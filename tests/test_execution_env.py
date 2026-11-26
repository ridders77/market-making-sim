"""Execution environment tests on hand-built mini books (prices in ticks of 0.1 USDT).

Mini setup unless stated: decisions every 0.3 s (3 buckets), horizon 0.9 s (3 steps), 0 latency,
pessimistic cancel rule, buy 0.5 BTC. Book: bid 1000 / ask 1001 (100.0 / 100.1 USDT), mid 100.05.
"""

import numpy as np
import pytest

from mms.execution.env import (BEHIND1, BUY, CROSS, SELL, TOUCH, ExecConfig, ExecutionEnv, run_execution,
                               shortfall_bps, taker_fill)
from tests_helpers_exec import N, flat_day, make_day


def cfg(**kw):
    base = dict(side=BUY, qty=0.5, horizon_s=0.9, decision_s=0.3, latency_ms=0, cancel_rule="pessimistic",
                maker_bps=2.0, taker_bps=5.0)
    base.update(kw)
    return ExecConfig(**base)


# ------------------------------------------------------------------ pure functions
def test_shortfall_hand_worked():
    # buy 0.5 BTC against m0 = 100: 0.2 at 99.95 as maker (2 bp), 0.3 at 100.05 as taker (5 bp)
    # USDT: 0.2*(-0.05) + 0.3*0.05 + 0.2*99.95*2e-4 + 0.3*100.05*5e-4 = 0.0240055; / 50 * 1e4 = 4.8011 bps
    c = shortfall_bps(BUY, [99.95, 100.05], [0.2, 0.3], [2.0, 5.0], 100.0, 0.5)
    assert c == pytest.approx(4.8011, abs=1e-9)
    # sell the same: receives p (1 - f). USDT: -(0.2*(-0.05) + 0.3*0.05) + fees = -0.005 + 0.0190055
    c = shortfall_bps(SELL, [99.95, 100.05], [0.2, 0.3], [2.0, 5.0], 100.0, 0.5)
    assert c == pytest.approx(1e4 * 0.0140055 / 50, abs=1e-9)
    # a maker rebate is a negative cost
    assert shortfall_bps(BUY, [100.0], [0.5], [-0.5], 100.0, 0.5) == pytest.approx(-0.5)


def test_taker_fill_beyond_displayed_size():
    fills, beyond = taker_fill(1.0, 1001, 0.4, BUY, depth_frac=0.5)
    assert beyond
    assert [p for p, _ in fills] == [1001, 1002, 1003, 1004]
    assert [x for _, x in fills] == pytest.approx([0.4, 0.2, 0.2, 0.2])
    fills, beyond = taker_fill(0.3, 1000, 0.4, SELL)
    assert fills == [(1000, 0.3)] and not beyond
    fills, _ = taker_fill(0.5, 1000, 0.4, SELL)
    assert fills[1] == (999, pytest.approx(0.1))      # a seller walks down


# ------------------------------------------------------------------ deadline and fees
def test_deadline_cleanup_buys_remainder_as_taker():
    # nothing trades: the passive order never fills; at t_end (bucket 10) the book in force is row 9
    ask = [1001] * N
    ask[9] = 1003                                     # the ask in force at the deadline
    day = make_day([1000] * N, ask, [2.0] * N, [3.0] * N)
    env = ExecutionEnv(day, cfg())
    env.reset(1)
    costs = []
    for _ in range(3):
        _, c, done, _ = env.step(TOUCH)
        costs.append(c)
    assert done
    r = env.result()
    assert r.cleanup_qty == pytest.approx(0.5) and r.passive_qty == 0.0 and r.taker_qty == pytest.approx(0.5)
    m0 = 100.05
    expect = 1e4 * (0.5 * (100.3 - m0) + 5e-4 * 0.5 * 100.3) / (0.5 * m0)
    assert r.cost_bps == pytest.approx(expect)
    assert sum(costs) == pytest.approx(expect)        # shaped step costs add up to the shortfall


def test_cleanup_beyond_displayed_size_is_counted_and_charged():
    day = make_day([1000] * N, [1001] * N, [2.0] * N, [0.2] * N)   # only 0.2 BTC shown at the ask
    r = run_execution(lambda o: TOUCH, day, 1, cfg())
    assert r.beyond_display == 1 and r.beyond_cleanup
    # 0.2 at 1001, then 0.1 at 1002, 1003, 1004
    px = r.fills["price"]
    assert list(px) == [1001, 1002, 1003, 1004]


def test_fee_accounting_passive_fill_and_repricing():
    # sellers hit 3 BTC at our bid in bucket 2: queue ahead 2.0, so 1.0 reaches us and fills all 0.5
    day = flat_day(trades=[(2, 1000, 3.0, -1)])
    r = run_execution(lambda o: TOUCH, day, 1, cfg())
    assert r.passive_qty == pytest.approx(0.5) and r.taker_qty == 0.0 and r.cleanup_qty == 0.0
    m0 = 100.05
    expect = 1e4 * (0.5 * (100.0 - m0) + 2e-4 * 0.5 * 100.0) / (0.5 * m0)
    assert r.cost_bps == pytest.approx(expect)
    # repricing to the professional fees equals a run at those fees
    r2 = run_execution(lambda o: TOUCH, day, 1, cfg(maker_bps=-0.5, taker_bps=2.0))
    assert r.cost_at(-0.5, 2.0) == pytest.approx(r2.cost_bps)
    assert r2.cost_bps < expect


def test_cross_pays_taker_fee_after_latency():
    ask = [1001] * N
    ask[1] = 1002                                      # row 1 = book in force at t_2 (arrival with 100 ms latency)
    day = make_day([1000] * N, ask, [2.0] * N, [3.0] * N)
    env = ExecutionEnv(day, cfg(latency_ms=100))
    env.reset(2)                                       # decide at t_2 (sees row 1), arrives at t_3 (row 2)
    r_env = env.step(CROSS)
    assert r_env[2]
    r = env.result()
    assert r.fills["price"][0] == 1001 and r.taker_qty == pytest.approx(0.5)
    env.reset(1)                                       # decide at t_1, arrives at t_2: row 1 -> 1002
    env.step(CROSS)
    assert env.result().fills["price"][0] == 1002


# ------------------------------------------------------------------ keep-queue rule
def _seller_flow():
    # sellers trade 1.0 BTC at our bid in buckets 2, 5 and 8 (one per step); the level shows 2.0 throughout
    return flat_day(trades=[(2, 1000, 1.0, -1), (5, 1000, 1.0, -1), (8, 1000, 1.0, -1)])


def test_unchanged_action_keeps_queue_place():
    env = ExecutionEnv(_seller_flow(), cfg())
    env.reset(1)
    env.step(TOUCH)                                    # joins with 2.0 ahead; bucket 2 eats 1.0
    assert env.order.Q == pytest.approx(1.0)
    env.step(TOUCH)                                    # same price: kept; bucket 5 eats the rest of the queue
    assert env.order.Q == pytest.approx(0.0)
    _, _, done, _ = env.step(TOUCH)                    # bucket 8: 1.0 traded, we are first -> 0.5 filled
    r = env.result()
    assert done and r.passive_qty == pytest.approx(0.5) and r.cleanup_qty == 0.0


def test_changing_action_rejoins_at_back():
    env = ExecutionEnv(_seller_flow(), cfg())
    env.reset(1)
    env.step(TOUCH)
    env.step(BEHIND1)                                  # cancel; behind the touch, queue unseen
    assert env.order.price == 999
    env.step(TOUCH)                                    # back to the touch: joins behind 2.0 again
    r = env.result()
    assert r.passive_qty == 0.0 and r.cleanup_qty == pytest.approx(0.5)


def test_unchanged_behind_order_kept_when_touch_falls_onto_it():
    # touch falls from 1000 to 999 at bucket 3 (bid level 999 shows 1.5); we sit 1 behind = 999 since step 1
    bid = [1000] * 3 + [999] * (N - 3)
    ask = [1001] * 3 + [1000] * (N - 3)
    day = make_day(bid, ask, [2.0] * 3 + [1.5] * (N - 3), [3.0] * N)
    env = ExecutionEnv(day, cfg())
    env.reset(1)
    env.step(BEHIND1)
    o = env.order
    assert o.price == 999 and o.Q == pytest.approx(1.5)   # level became best: Q capped by its size
    env.step(BEHIND1)                                  # target now 998, but the order is closer: kept
    assert env.order is o and o.price == 999
    env.step(TOUCH)                                    # new action, but its price (999) is ours: kept
    assert env.order is None or env.order.price == 999
    r = env.result()
    assert r.n_post_reject == 0


def test_market_moving_away_repegs_touch_order():
    bid = [1000] * 4 + [1001] * (N - 4)
    ask = [1001] * 4 + [1002] * (N - 4)
    day = make_day(bid, ask, [2.0] * N, [3.0] * N)
    env = ExecutionEnv(day, cfg())
    env.reset(1)
    env.step(TOUCH)
    assert env.order.price == 1000
    env.step(TOUCH)                                    # decision at t_4 sees row 3 (1000): kept
    assert env.order.price == 1000
    env.step(TOUCH)                                    # decision at t_7 sees 1001: the order re-pegs
    assert env.result().fills["price"][0] == 1002      # nothing filled; clean-up at the ask 1002


def test_sell_side_mirror():
    # buyers lift 3.0 at our ask 1001 in bucket 2: queue 3.0 ahead -> nothing; bucket 5 another 1.0 -> 0.5 fill
    day = flat_day(trades=[(2, 1001, 3.0, 1), (5, 1001, 1.0, 1)])
    r = run_execution(lambda o: TOUCH, day, 1, cfg(side=SELL))
    assert r.passive_qty == pytest.approx(0.5)
    m0 = 100.05
    assert r.cost_bps == pytest.approx(1e4 * (-(0.5 * (100.1 - m0)) + 2e-4 * 0.5 * 100.1) / (0.5 * m0))


def test_exploring_start_cost_in_bps_of_full_order():
    env = ExecutionEnv(flat_day(), cfg())
    env.reset(4, steps_left=1, remaining=0.1)
    _, c, done, _ = env.step(CROSS)
    assert done
    assert c == pytest.approx(1e4 * (0.1 * (100.1 - 100.05) + 5e-4 * 0.1 * 100.1) / (0.5 * 100.05))


def test_twap_slices_add_up_and_pay_taker():
    from mms.execution.baselines import TWAP
    tw = TWAP(0.5, 6)
    assert [tw.slice_lots(k) for k in range(6)] == [8, 9, 8, 8, 9, 8] and sum(tw.slice_lots(k) for k in range(6)) == 50
    tw3 = TWAP(0.5, 3)
    r = run_execution(tw3, flat_day(), 1, cfg())
    assert r.taker_qty == pytest.approx(0.5) and r.passive_qty == 0.0 and r.cleanup_qty == 0.0
    assert r.cost_bps == pytest.approx(1e4 * (0.5 * 0.05 + 5e-4 * 0.5 * 100.1) / (0.5 * 100.05))
