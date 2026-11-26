"""The stage 5 RL API must reproduce the stage 4 baselines exactly."""

import numpy as np
import pytest

import mms.replay.rl_env as rl
from mms.replay.data import STAGE4
from mms.replay.env import ReplayConfig, run_episode
from mms.strategies.touch import AlwaysTouch, TouchOFIAware, TouchQueueAware, TouchSkew
from test_replay_env import make_day


def _random_day(seed=5, n=700):
    rng = np.random.default_rng(seed)
    bid = 1000 + np.cumsum(rng.integers(-1, 2, n))
    ask = bid + rng.choice([1, 1, 1, 2], n)
    bq, aq = rng.uniform(0.05, 2.0, n), rng.uniform(0.05, 2.0, n)
    trades = []
    for b in range(1, n):
        for _ in range(rng.integers(0, 4)):
            s = int(rng.choice([-1, 1]))
            trades.append((b, int(bid[b - 1] - rng.integers(0, 2) if s < 0 else ask[b - 1] + rng.integers(0, 2)),
                           float(rng.uniform(0.01, 0.8)), s))
    d = make_day(bid, ask, bq, aq, trades=trades, bq_min=bq * 0.6, aq_min=aq * 0.6)
    d.ofi_z = rng.normal(0, 1, n)
    d.typical_size = np.full(n, 1.0)
    return d


def discrete(policy):
    """Express a touch/off baseline as one of the 16 discrete actions."""
    def f(obs):
        raw = obs["raw"]
        b, a = policy(raw)
        return rl.encode_action("off" if b is None else "touch", "off" if a is None else "touch")
    return f


@pytest.mark.parametrize("make", [AlwaysTouch, lambda: TouchOFIAware(0.8), lambda: TouchQueueAware(0.7),
                                  lambda: TouchSkew(3, 1e-2)])
def test_baselines_identical_through_api(monkeypatch, make):
    day = _random_day()
    monkeypatch.setattr(rl, "load_day", lambda d: day)
    cfg = ReplayConfig(latency_ms=100, cancel_rule="proportional", episode_s=60.0, gamma=1e-4)
    ref = run_episode(make(), day, 5, cfg)
    assert ref.n_fills > 5
    env = rl.ReplayRLEnv(split="x", config=cfg, days=["test"])
    obs = env.reset(day="test", start=5)
    pol = make()
    use_discrete = not isinstance(pol, TouchSkew)          # the skew needs arbitrary tick offsets
    act = discrete(pol) if use_discrete else (lambda o: pol(o["raw"]))
    done, total = False, 0.0
    while not done:
        obs, rew, done, info = env.step(act(obs))
        total += rew
    r = env.result()
    assert r.pnl == ref.pnl and r.fill_qty == ref.fill_qty and r.q == ref.q
    assert np.array_equal(r.fills["step"], ref.fills["step"])
    assert total == pytest.approx(r.pnl - r.penalty, abs=1e-9)


def test_action_encoding():
    assert rl.decode_action(0) == (0, 0)
    assert rl.decode_action(rl.encode_action("3 behind", "off")) == (3, None)
    raw = {"best_bid": 1000, "best_ask": 1001}
    assert rl.action_to_prices(rl.encode_action("1 behind", "3 behind"), raw) == (999, 1004)
    assert len({rl.decode_action(a) for a in range(16)}) == 16
    with pytest.raises(ValueError):
        rl.decode_action(16)


def test_seed_picks_split_day_and_valid_start():
    env = rl.ReplayRLEnv(split="calibration", days=["2023-05-30", "2023-06-15"])
    for seed in range(50):
        day, start = env.episode_start(seed)
        assert day in env.days
        assert 36000 <= start <= 864000 - 36000
    assert env.episode_start(7) == env.episode_start(7)
    assert set(rl.split_days("calibration")).isdisjoint(rl.split_days("test"))


@pytest.mark.skipif(not (STAGE4 / "BTCUSDT-book100ms-2023-05-30.parquet").exists(), reason="stage 4 data not built")
def test_real_day_observation():
    env = rl.ReplayRLEnv(split="calibration", days=["2023-05-30"])
    obs = env.reset(seed=1)
    assert obs["vector"].shape == (7,) and np.all(np.isfinite(obs["vector"][[0, 1, 4, 6]]))
    obs, rew, done, _ = env.step(0)
    assert np.isfinite(rew) and not done
