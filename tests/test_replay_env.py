"""Replay backtester tests on hand-built mini books (prices in ticks of 0.1 USDT)."""

import numpy as np
import pytest

from mms.replay.data import ReplayDay
from mms.replay.env import ReplayConfig, ReplayEnv, inventory_penalty, run_episode
from mms.replay.queue import BID


def make_day(bid, ask, bq, aq, trades=(), bq_min=None, aq_min=None, sigma=1.0) -> ReplayDay:
    """Rows = book at the end of each bucket. trades: (bucket, price_tick, qty, side)."""
    bid, ask = np.asarray(bid, np.int64), np.asarray(ask, np.int64)
    bq, aq = np.asarray(bq, float), np.asarray(aq, float)
    nb = len(bid)
    bq_min = bq.copy() if bq_min is None else np.asarray(bq_min, float)
    aq_min = aq.copy() if aq_min is None else np.asarray(aq_min, float)
    tr = sorted(trades)
    b = np.array([t[0] for t in tr], dtype=np.int64)
    p = np.array([t[1] for t in tr], dtype=np.int64)
    q = np.array([t[2] for t in tr], dtype=float)
    s = np.array([t[3] for t in tr], dtype=np.int8)
    big = 10**15
    sell_min, buy_max = np.full(nb, big), np.full(nb, -big)
    for bb, pp, ss in zip(b, p, s):
        if ss < 0:
            sell_min[bb] = min(sell_min[bb], pp)
        else:
            buy_max[bb] = max(buy_max[bb], pp)
    return ReplayDay(day="test", t_ms=np.arange(1, nb + 1) * 100, bid=bid, ask=ask, bid_qty=bq, ask_qty=aq,
                     bid_qty_min=bq_min, ask_qty_min=aq_min, mid=(bid + ask) * 0.1 / 2,
                     tr_start=np.searchsorted(b, np.arange(nb + 1)), tr_price=p, tr_qty=q, tr_side=s,
                     sell_min=sell_min, buy_max=buy_max, ofi_z=np.zeros(nb), sigma_hat=np.full(nb, sigma),
                     typical_size=np.ones(nb))


def cfg(**kw):
    base = dict(latency_ms=0, cancel_rule="pessimistic", lot=0.01, episode_s=1.0, gamma=0.0)
    base.update(kw)
    return ReplayConfig(**base)


class Fixed:
    def __init__(self, bid=None, ask=None):
        self.bid, self.ask = bid, ask

    def __call__(self, obs):
        return self.bid, self.ask


def test_fill_exactly_when_traded_volume_exceeds_queue_in_replay():
    n = 12
    # touch 1000 / 1001; bid level shows 2.0 BTC; sellers trade 1.5 at 1000 in bucket 3, 0.6 in bucket 6
    bq = [2.0] * 3 + [0.5 + 1.0] * 3 + [0.9 + 1.0] * 6      # level replenished behind us after each trade
    bq_min = [2.0] * 3 + [0.5] + [1.5] * 2 + [0.9] + [1.9] * 5
    day = make_day([1000] * n, [1001] * n, bq, [3.0] * n, trades=[(3, 1000, 1.5, -1), (6, 1000, 0.6, -1)],
                   bq_min=bq_min)
    env = ReplayEnv(day, cfg())
    obs = env.reset(1)
    fills = {}
    done = False
    while not done:
        i = env.i
        obs, _, done, info = env.step((1000, None))
        if info["fills"]:
            fills[i] = info["fills"]
    # joined at step 1 behind 2.0 BTC; 1.5 traded -> 0.5 ahead; 0.6 traded -> 0.1 beyond the queue: full lot
    assert list(fills) == [6]
    side, price, qty, how = fills[6][0]
    assert (side, price, how) == (BID, 1000, "queue") and qty == pytest.approx(0.01)
    r = env.result()
    assert r.q == pytest.approx(0.01) and r.notional == pytest.approx(0.01 * 100.0)


@pytest.mark.parametrize("latency_ms, first_fill_step", [(0, 1), (100, 2), (300, 4)])
def test_latency_delays_activation(latency_ms, first_fill_step):
    n = 12
    # 3-tick spread 999 / 1002; our bid at 1000 improves the best bid (alone there); a seller prints at 999
    # (through our price) in every bucket from 1 on
    day = make_day([999] * n, [1002] * n, [1.0] * n, [1.0] * n, trades=[(b, 999, 0.5, -1) for b in range(1, n)])
    env = ReplayEnv(day, cfg(latency_ms=latency_ms))
    env.reset(1)
    steps = []
    done = False
    while not done:
        i = env.i
        _, _, done, info = env.step((1000, None))
        if info["fills"]:
            steps.append(i)
    assert steps[0] == first_fill_step
    assert env.result().fills["how"][0] == "through"


def _cancel_day():
    """We join bid level 1000 behind 2 BTC; 8 BTC join behind us; then 8 BTC are cancelled (2 per bucket in
    buckets 2-5) and sellers trade 0.2 (bucket 7) and 0.1 (bucket 8) at 1000."""
    n = 12
    bq = [2.0, 10.0, 8.0, 6.0, 4.0, 2.0, 2.0, 1.8, 1.7, 1.7, 1.7, 1.7]
    trades = [(7, 1000, 0.2, -1), (8, 1000, 0.1, -1)]
    return make_day([1000] * n, [1001] * n, bq, [5.0] * n, trades=trades)


def test_cancel_rules_give_expected_fill_ordering():
    """Hand-worked queue ahead after the cancels:
    optimistic: cancels come from ahead first -> 0, so the 0.2 trade fills our lot;
    proportional: each 2 BTC cancel removes 2 * Q / level: 2 -> 1.6 -> 1.2 -> 0.8 -> 0.4; trades 0.3 < 0.4: no fill;
    pessimistic: no cancels ahead, size never below 2 while cancelling -> 2; trades 0.3: no fill."""
    out, Q = {}, {}
    for rule in ("pessimistic", "proportional", "optimistic"):
        env = ReplayEnv(_cancel_day(), cfg(cancel_rule=rule, episode_s=1.1))
        env.reset(1)
        for _ in range(6):                      # steps 1..6: join, then the cancels of buckets 2-5
            env.step((1000, None))
        Q[rule] = env.orders[BID].Q
        done = False
        while not done:
            _, _, done, _ = env.step((1000, None))
        out[rule] = env.result().fill_qty
    assert Q["optimistic"] == pytest.approx(0.0)
    assert Q["proportional"] == pytest.approx(0.4)
    assert Q["pessimistic"] == pytest.approx(2.0)
    assert out == pytest.approx({"optimistic": 0.01, "proportional": 0.0, "pessimistic": 0.0})
    assert out["optimistic"] >= out["proportional"] >= out["pessimistic"]


def test_accounting_conserves_value_and_fee_repricing_is_exact():
    rng = np.random.default_rng(3)
    n = 400
    mid_path = 1000 + np.cumsum(rng.integers(-1, 2, n))
    bid, ask = mid_path, mid_path + 1
    bq, aq = rng.uniform(0.01, 0.5, n), rng.uniform(0.01, 0.5, n)
    trades = []
    for b in range(1, n):
        for _ in range(rng.integers(0, 3)):
            s = rng.choice([-1, 1])
            trades.append((b, int(bid[b - 1] if s < 0 else ask[b - 1]), float(rng.uniform(0.01, 0.3)), int(s)))
    day = make_day(bid, ask, bq, aq, trades=trades, bq_min=bq * 0.5, aq_min=aq * 0.5)
    touch = lambda o: (o["best_bid"], o["best_ask"])
    for fee in (0.0, 2.0, -0.5):
        env = ReplayEnv(day, cfg(episode_s=39.0, fee_bps=fee, cancel_rule="proportional"))
        obs = env.reset(1)
        total, done = 0.0, False
        while not done:
            obs, rew, done, _ = env.step(touch(obs))
            total += rew
        r = env.result()
        assert r.n_fills > 5
        mid_end = day.mid[env.end - 1]
        f = r.fills
        px = f["price"] * 0.1
        # P&L = sum of rewards = cash + q * final mid = sum over fills of side * qty * (mid_end - price) - fees
        assert total == pytest.approx(r.pnl, abs=1e-9)
        assert r.pnl == pytest.approx(r.cash + r.q * mid_end, abs=1e-9)
        by_fill = np.sum(f["side"] * f["qty"] * (mid_end - px)) - fee * 1e-4 * np.sum(f["qty"] * px)
        assert r.pnl == pytest.approx(by_fill, abs=1e-9)
        assert r.q == pytest.approx(np.sum(f["side"] * f["qty"]), abs=1e-12)
        if fee == 0.0:
            r0 = r
        else:
            assert r0.pnl_with_fee(fee) == pytest.approx(r.pnl, abs=1e-9)


def test_inventory_limit_and_post_only():
    n = 30
    day = make_day([999] * n, [1002] * n, [1.0] * n, [1.0] * n, trades=[(b, 999, 0.5, -1) for b in range(1, n)])
    r = run_episode(Fixed(bid=1000), day, 1, cfg(lot=0.01, inv_limit_lots=3, episode_s=2.5))
    assert r.q == pytest.approx(0.03) and r.max_abs_q == pytest.approx(0.03)
    assert r.n_rejected_limit > 0
    r = run_episode(Fixed(bid=1002), day, 1, cfg(episode_s=1.0))          # bid at the best ask: rejected
    assert r.n_fills == 0 and r.n_rejected_post_only == 10


def test_inventory_penalty_formula():
    # 3 lots of 0.01 BTC, gamma 1e-4 per lot, sigma 2 USDT/sqrt(s), dt 0.1 s: 0.5 * 1e-4 * 0.01 * 9 * 4 * 0.1
    assert inventory_penalty(0.03, 0.01, 1e-4, 2.0, 0.1) == pytest.approx(1.8e-6)


def test_keep_place_when_price_unchanged_and_rejoin_when_moved():
    n = 12
    day = make_day([1000] * n, [1001] * n, [4.0] * n, [4.0] * n)
    env = ReplayEnv(day, cfg())
    env.reset(1)
    env.step((1000, None))
    env.orders[BID].Q = 1.0                     # pretend we moved up the queue
    env.step((1000, None))
    assert env.orders[BID].Q == 1.0             # same price: same order, same place
    env.step((999, None))
    assert env.orders[BID].price == 999 and env.orders[BID].Q == float("inf")   # moved behind the touch


def test_queue_frac_what_if():
    """Stage 5 Part B: joining with only a fraction f of the displayed queue ahead (f = 1 is stage 4)."""
    n = 6
    day = make_day([1000] * n, [1001] * n, [2.0] * n, [3.0] * n, trades=[(2, 1000, 0.5, -1)])
    for f, filled in ((1.0, 0.0), (0.5, 0.0), (0.2, 0.01), (0.0, 0.01)):
        env = ReplayEnv(day, cfg(queue_frac=f, episode_s=0.4))
        env.reset(1)
        env.step((1000, None))
        assert env.orders[BID] is None or env.orders[BID].Q == pytest.approx(max(0.0, 2.0 * f))
        env.step((1000, None))
        env.step((1000, None))
        env.step((1000, None))
        assert env.result().fill_qty == pytest.approx(filled)
    with pytest.raises(ValueError):
        cfg(queue_frac=1.5)
