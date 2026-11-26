"""Tabular Q-learning pieces: state buckets, the update rule, greedy evaluation, epsilon schedule."""

import numpy as np
import pytest

from mms.execution.env import BUY, CROSS, SELL, TOUCH, ExecConfig, ExecutionEnv
from mms.execution.qlearning import GreedyPolicy, StateSpec, TabularQ, epsilon
from tests_helpers_exec import flat_day


def test_time_and_inventory_buckets():
    sp = StateSpec()
    # N = 6 steps: steps_left 6 and 5 share the first bucket; the last decision has its own
    assert [sp.time_bucket(k, 6) for k in (6, 5, 4, 3, 2, 1)] == [4, 4, 3, 2, 1, 0]
    assert [sp.time_bucket(k, 30) for k in (30, 25, 24, 7, 6, 1)] == [4, 4, 3, 1, 0, 0]
    # remaining fraction: (0, .2] -> 0, (.2, .4] -> 1, ..., (.8, 1] -> 4
    assert [sp.inv_bucket(x) for x in (1.0, 0.81, 0.8, 0.41, 0.4, 0.2, 0.02)] == [4, 4, 3, 2, 1, 0, 0]


def test_ofi_signed_towards_our_side():
    sp = StateSpec(market=("ofi",), ofi_c=0.4)
    assert sp.ofi_bucket(BUY, 1.0) == 2          # buyers dominate: adverse for a buyer
    assert sp.ofi_bucket(SELL, 1.0) == 0         # ... favourable for a seller
    assert sp.ofi_bucket(BUY, -0.5) == 0
    assert sp.ofi_bucket(BUY, 0.3) == 1
    assert sp.ofi_bucket(BUY, float("nan")) == 1  # no trades in 5 s: no imbalance


def test_encode_all_variables():
    sp = StateSpec(market=("ofi", "queue", "sigma"), ofi_c=0.4, queue_cuts=(0.5, 1.5), sigma_cuts=(2.0, 3.0))
    obs = {"steps_left": 6, "n_steps": 6, "remaining_frac": 0.5, "side": BUY, "ofi_z": 0.9,
           "own_size": 2.0, "typical_size": 1.0, "sigma_hat": 2.5}
    assert sp.encode(obs) == (4, 2, 2, 2, 1)
    assert sp.shape == (5, 5, 3, 3, 3)


def test_update_rule_hand_worked():
    ag = TabularQ((2,))
    ag.Q[1] = [3.0, 1.0, 2.0, 5.0]
    # target = 0.5 + min Q(s') = 1.5; Q(0, touch) = 0 + 0.1 * 1.5
    ag.update((0,), TOUCH, 0.5, (1,), alpha=0.1)
    assert ag.Q[0, TOUCH] == pytest.approx(0.15) and ag.N[0, TOUCH] == 1
    ag.update((0,), TOUCH, 2.0, None, alpha=0.5)            # terminal: target = cost
    assert ag.Q[0, TOUCH] == pytest.approx(0.15 + 0.5 * (2.0 - 0.15))


def test_greedy_ignores_untried_actions_and_counts_fallbacks():
    ag = TabularQ((2, 1))
    ag.Q[0, 0] = [4.0, 0.0, 0.0, 0.0]
    ag.N[0, 0] = [1, 0, 0, 0]                    # only "cross" tried: chosen although Q = 0 elsewhere
    assert ag.greedy((0, 0)) == CROSS and ag.fallbacks == 0
    # 3-dim table: a market cell never visited falls back to the (time, inv) average
    ag = TabularQ((1, 1, 2))
    ag.Q[0, 0, 0] = [4.0, 1.0, 9.0, 9.0]
    ag.N[0, 0, 0] = [5, 5, 1, 1]
    assert ag.greedy((0, 0, 1)) == TOUCH and ag.fallbacks == 1
    assert ag.policy_table().tolist() == [[[TOUCH, -1]]]


def test_epsilon_schedule():
    assert epsilon(0, 1000, 0.5) == 1.0
    assert epsilon(250, 1000, 0.5) == pytest.approx(1 - 0.95 * 0.5)
    assert epsilon(500, 1000, 0.5) == 0.05 and epsilon(999, 1000, 0.5) == 0.05


def test_learns_to_rest_when_passive_fills_are_certain():
    # sellers trade 3 BTC at the bid every bucket: resting at the touch always fills at 2 bp maker fee,
    # crossing costs the half-spread + 5 bp. The agent should learn "touch" at the start state.
    trades = [(j, 1000, 3.0, -1) for j in range(1, 14)]
    day = flat_day(trades=trades)
    cfg = ExecConfig(side=BUY, qty=0.5, horizon_s=0.9, decision_s=0.3, latency_ms=0, cancel_rule="pessimistic")
    sp = StateSpec()
    ag = TabularQ(sp.shape)
    rng = np.random.default_rng(0)
    env = ExecutionEnv(day, cfg)
    for ep in range(300):
        obs = env.reset(1)
        done = False
        while not done:
            s = sp.encode(obs)
            a = ag.act(s, epsilon(ep, 300, 0.5), rng)
            obs, c, done, _ = env.step(a)
            ag.update(s, a, c, None if done else sp.encode(obs), alpha=0.2)
    assert GreedyPolicy(ag, sp)(env.reset(1)) == TOUCH
    assert ag.Q[4, 4, CROSS] > ag.Q[4, 4, TOUCH]
