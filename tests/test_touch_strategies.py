"""Hand-worked checks of the stage 4 touch strategies."""

import math

import numpy as np
import pytest

from mms.strategies.touch import AlwaysTouch, ReplayMarkoutAwareAS, TouchOFIAware, TouchQueueAware, TouchSkew


def obs(**kw):
    base = {"best_bid": 1000, "best_ask": 1001, "bid_size": 2.0, "ask_size": 8.0, "typical_size": 10.0,
            "inventory": 0.0, "sigma_hat": 2.0, "time_remaining": 1800.0, "t": 1800.0, "ofi_z": 0.0,
            "bid_order": None, "ask_order": None, "mid": 100.05}
    base.update(kw)
    return base


def test_always_touch():
    assert AlwaysTouch()(obs()) == (1000, 1001)


def test_skew_ticks_hand_worked():
    s = TouchSkew(max_lots=5, gamma=1e-4)
    # q = 2 lots: 2 * 1e-4 * 4 * 1800 / 0.1 = 14.4 -> 14 ticks
    assert s.skew_ticks(obs(inventory=2.0)) == 14
    # long: bid 14 ticks behind; ask cannot improve inside a 1-tick spread
    assert s(obs(inventory=2.0)) == (986, 1001)
    # long with a 5-tick spread: ask improves to the bid + 1 at most
    assert s(obs(inventory=2.0, best_ask=1005)) == (986, 1001)
    # short 0.5 lot: n = -3.6 -> -4; ask 4 behind, bid improves by up to 4 inside a 10-tick spread
    assert s(obs(inventory=-0.5, best_ask=1010)) == (1004, 1014)
    # threshold: at +5 lots the bid is pulled
    assert s(obs(inventory=5.0))[0] is None
    assert TouchSkew(max_lots=2, gamma=0.0)(obs(inventory=-2.0)) == (1000, None)


def test_queue_aware():
    s = TouchQueueAware(theta=0.5)
    assert s(obs()) == (1000, None)                        # bid queue 2 < 5, ask queue 8 >= 5
    assert s(obs(ask_order=(1001, 3.0))) == (1000, 1001)   # already at the touch: keep our place
    assert s(obs(ask_order=(1002, 3.0))) == (1000, None)   # our old order is not at the touch: do not rejoin
    assert s(obs(typical_size=float("nan"))) == (None, None)


def test_ofi_aware():
    s = TouchOFIAware(z_star=1.0)
    assert s(obs(ofi_z=1.2)) == (1000, None)               # buyers aggressive: pull the ask
    assert s(obs(ofi_z=-1.2)) == (None, 1001)
    assert s(obs(ofi_z=0.5)) == (1000, 1001)
    assert s(obs(ofi_z=float("nan"))) == (1000, 1001)


def test_markout_as_quotes_on_ticks_and_away_from_mid():
    mb = {"a": 0.08, "b": 1.35, "c": 0.64, "d": -0.38}
    s = ReplayMarkoutAwareAS(1e-4, mb, jump_delta=[0.05, 1.0, 5.0], jump_mean_bps=[0.1, 1.0, 1.2], fee_bps=0.0)
    o = obs(mid=30000.05, t=0.0, time_remaining=3600.0, sigma_hat=1.5)
    s.reset(o)
    b, a = s(o)
    assert isinstance(b, int) and isinstance(a, int)
    assert b * 0.1 <= 30000.05 - 0.1 + 1e-9 and a * 0.1 >= 30000.05 + 0.1 - 1e-9
    assert s.p["k"] == pytest.approx(0.64 * 1.5 ** -0.38)
    assert s.p["jump_mean"][1] == pytest.approx(1.0 * 30000.05 / 1e4)
