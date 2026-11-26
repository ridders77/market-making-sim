"""Stage 3 estimator tests: lot-size touch excursions and the trailing EWMA sigma."""

import numpy as np
import pandas as pd
import pytest

from mms.calibration.estimators import lot_excursions, touch_excursions, trailing_sigma


def _one_window():
    # One 1-second window, mid0 = 100.0. Buyer-initiated prints (lift the ask):
    #   100.05 x 0.004, 100.15 x 0.003, 100.25 x 0.005
    # Seller-initiated prints (hit the bid): 99.95 x 0.020
    snap = pd.DataFrame({"time_s": [0], "mid": [100.0]})
    trades = pd.DataFrame({
        "time": [100, 200, 300, 400],
        "price": [100.05, 100.15, 100.25, 99.95],
        "qty": [0.004, 0.003, 0.005, 0.020],
        "is_buyer_maker": [False, False, False, True],
    })
    return snap, trades


@pytest.mark.parametrize("lot, expected_ask", [
    # Volume printed at or through distance d (cumulated from the deepest print inwards):
    #   d <= 0.25: 0.005;  d <= 0.15: 0.005 + 0.003 = 0.008;  d <= 0.05: 0.008 + 0.004 = 0.012
    (0.005, 0.25),   # the 0.25 print alone fills a 0.005 lot
    (0.007, 0.15),   # needs the 0.25 and 0.15 prints together
    (0.010, 0.05),   # needs all three: only an order at the touch (0.05) gets 0.01 BTC
    (0.012, 0.05),   # exactly the total (cumulated with a float tolerance)
    (0.020, -np.inf),  # not enough buying volume in the window: no ask fill at any distance
])
def test_lot_excursion_hand_worked(lot, expected_ask):
    snap, trades = _one_window()
    w = lot_excursions(snap, trades, window_s=1, lot=lot)
    assert w.ask_exc.iloc[0] == pytest.approx(expected_ask)
    # bid side: one 0.020 sell at 99.95, i.e. 0.05 below the mid: enough for every lot here
    assert w.bid_exc.iloc[0] == pytest.approx(0.05)


def test_lot_excursion_with_tiny_lot_equals_any_trade_rule():
    # As the lot -> 0, "at least one lot traded at or through d" becomes "any trade at or through d",
    # which is the stage 2 touch excursion (the extreme print in the window).
    rng = np.random.default_rng(0)
    n = 4000
    snap = pd.DataFrame({"time_s": np.arange(60), "mid": 30000 + np.round(rng.normal(0, 2, 60), 1) + 0.05})
    trades = pd.DataFrame({
        "time": np.sort(rng.integers(0, 60_000, n)),
        "price": np.round(30000 + rng.normal(0, 3, n), 1),
        "qty": rng.choice([0.001, 0.01, 0.2, 1.5], n),
        "is_buyer_maker": rng.random(n) < 0.5,
    })
    a = lot_excursions(snap, trades, window_s=1, lot=1e-9)
    b = touch_excursions(snap, trades, window_s=1)
    np.testing.assert_allclose(a.ask_exc, b.ask_exc)
    np.testing.assert_allclose(a.bid_exc, b.bid_exc)
    # A bigger lot can only make the fill distance smaller (or remove the fill)
    c = lot_excursions(snap, trades, window_s=1, lot=0.5)
    assert (c.ask_exc <= a.ask_exc + 1e-12).all() and (c.bid_exc <= a.bid_exc + 1e-12).all()


def test_lot_excursion_windows_use_their_own_mid():
    # Two windows with different mids; the same print is a different distance from each.
    snap = pd.DataFrame({"time_s": [0, 1], "mid": [100.0, 100.2]})
    trades = pd.DataFrame({"time": [500, 1500], "price": [100.3, 100.3], "qty": [0.01, 0.01],
                           "is_buyer_maker": [False, False]})
    w = lot_excursions(snap, trades, window_s=1, lot=0.01)
    np.testing.assert_allclose(w.ask_exc, [0.3, 0.1])
    assert np.isneginf(w.bid_exc).all()


# ---------------------------------------------------------------- trailing sigma
def _minute_snap(changes, start_mid=100.0):
    """1-second snapshots where the mid jumps by changes[j] at the start of minute j+1."""
    mids = start_mid + np.concatenate([[0.0], np.cumsum(changes)])
    return pd.DataFrame({"time_s": np.arange(len(mids) * 60), "mid": np.repeat(mids, 60)})


def test_trailing_sigma_constant_changes():
    # Every 60 s change is +-3 USDT, so (dS)^2 / 60 = 9/60 per second and any EWMA of it equals 9/60:
    # sigma_hat = sqrt(9/60) = 0.3873 USDT/sqrt(s), once 30 changes have been seen.
    changes = 3.0 * np.where(np.arange(50) % 2 == 0, 1, -1)
    snap = _minute_snap(changes)
    s = trailing_sigma(snap, span_min=30)
    assert np.isnan(s[: 30 * 60]).all()           # first 30 minutes: not enough history, NaN (not filled)
    assert s[30 * 60] == pytest.approx(np.sqrt(9 / 60))
    assert s[-1] == pytest.approx(np.sqrt(9 / 60))


def test_trailing_sigma_uses_only_the_past():
    # Changing mids after second t must not change sigma_hat at t.
    rng = np.random.default_rng(3)
    changes = rng.normal(0, 2, 90)
    a = trailing_sigma(_minute_snap(changes))
    changes2 = changes.copy()
    changes2[60:] *= 10                             # the change that ends at minute 61 onwards
    b = trailing_sigma(_minute_snap(changes2))
    t = 61 * 60                                     # first second that may see change #61 (index 60)
    np.testing.assert_array_equal(a[:t], b[:t])
    assert b[t] > a[t]


def test_trailing_sigma_ewma_weights_hand_worked():
    # span 3 -> alpha = 2/(3+1) = 0.5, pandas adjust=True weights (1, 1-a, (1-a)^2, ...) normalised.
    # Squared changes / 60: c = [60, 0, 120]/60 = [1, 0, 2] for changes sqrt(60), 0, sqrt(120).
    # After 3 changes: (2*1 + 0*0.5 + 1*0.25) / (1 + 0.5 + 0.25) = 2.25 / 1.75 = 1.285714
    snap = _minute_snap(np.array([np.sqrt(60), 0.0, np.sqrt(120)]))
    s = trailing_sigma(snap, span_min=3)
    assert s[3 * 60] == pytest.approx(np.sqrt(2.25 / 1.75))
    assert np.isnan(s[3 * 60 - 1])
