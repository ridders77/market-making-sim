"""Hand-worked tests for the queue model (one resting order, one 100 ms bucket at a time).

Prices are integer ticks (1 tick = 0.1 USDT). BID = our buy order (filled by seller-initiated trades),
ASK = our sell order (filled by buyer-initiated trades). Trade side: +1 buyer-initiated, -1 seller-initiated.
"""

import math

import numpy as np
import pytest

from mms.replay.queue import ASK, BID, BucketBook, Order, cancels_ahead, update_order

INF = math.inf


def book(best_prev, size_prev, best_end, size_min, opp_end):
    return BucketBook(best_prev=best_prev, size_prev=size_prev, best_end=best_end, size_min=size_min, opp_end=opp_end)


def trades(*rows):
    """rows of (price_tick, qty, side)."""
    if not rows:
        return np.array([], dtype=np.int64), np.array([]), np.array([], dtype=np.int8)
    p, q, s = zip(*rows)
    return np.array(p, dtype=np.int64), np.array(q, float), np.array(s, dtype=np.int8)


# ------------------------------------------------------------------ cancellation split
def test_cancel_rules_hand_worked():
    # 3 BTC cancelled at a level of 10 BTC (after trades), we have 4 ahead of us
    assert cancels_ahead("pessimistic", C=3.0, Q=4.0, level=10.0) == 0.0
    assert cancels_ahead("proportional", C=3.0, Q=4.0, level=10.0) == pytest.approx(1.2)   # 3 * 4/10
    assert cancels_ahead("optimistic", C=3.0, Q=4.0, level=10.0) == 3.0
    assert cancels_ahead("optimistic", C=6.0, Q=4.0, level=10.0) == 4.0                    # capped at Q
    assert cancels_ahead("proportional", C=3.0, Q=4.0, level=0.0) == 3.0                    # Q/level capped at 1
    with pytest.raises(ValueError):
        cancels_ahead("random", 1.0, 1.0, 1.0)


# ------------------------------------------------------------------ fills from traded volume
def test_fill_exactly_when_traded_volume_exceeds_queue_ahead():
    """Mini book: we bid at 1000 with 2.0 BTC ahead of us, 0.01 BTC order. Sellers trade at 1000."""
    bk = book(best_prev=1000, size_prev=5.0, best_end=1000, size_min=3.0, opp_end=1001)
    o = Order(side=BID, price=1000, qty=0.01, Q=2.0)
    # 1.5 BTC traded at our price: all of it eats the queue ahead, no fill, 0.5 left ahead
    r = update_order(o, *trades((1000, 1.5, -1)), bk, "pessimistic")
    assert r.filled == 0.0 and o.Q == pytest.approx(0.5)
    # exactly 0.5 more: queue ahead reaches 0, still no fill (volume must EXCEED the queue ahead)
    bk2 = book(1000, 3.0, 1000, 2.5, 1001)
    r = update_order(o, *trades((1000, 0.5, -1)), bk2, "pessimistic")
    assert r.filled == 0.0 and o.Q == 0.0
    # 0.004 more: we are first in line, partial fill of 0.004
    r = update_order(o, *trades((1000, 0.004, -1)), book(1000, 2.5, 1000, 2.496, 1001), "pessimistic")
    assert r.filled == pytest.approx(0.004) and o.qty == pytest.approx(0.006)
    # buyer-initiated trades never fill a bid
    r = update_order(o, *trades((1001, 9.0, 1)), book(1000, 2.5, 1000, 2.5, 1001), "pessimistic")
    assert r.filled == 0.0


def test_fill_in_one_bucket_with_volume_beyond_queue():
    o = Order(side=ASK, price=1001, qty=0.01, Q=1.0)
    bk = book(best_prev=1001, size_prev=3.0, best_end=1001, size_min=1.995, opp_end=1000)
    r = update_order(o, *trades((1001, 0.6, 1), (1001, 0.405, 1)), bk, "pessimistic")
    assert r.filled == pytest.approx(0.005) and o.qty == pytest.approx(0.005) and o.Q == 0.0


def test_trade_through_fills_fully():
    o = Order(side=BID, price=1000, qty=0.01, Q=50.0)
    # a seller printed at 999, below our bid: everything at 1000 is gone, so we are filled in full
    r = update_order(o, *trades((1000, 0.1, -1), (999, 0.2, -1)), book(1000, 60, 999, 1.0, 1000), "pessimistic")
    assert r.filled == pytest.approx(0.01) and r.how == "through" and o.qty == 0.0
    o = Order(side=ASK, price=1005, qty=0.01, Q=INF)       # 5 ticks behind the touch, depth unseen
    r = update_order(o, *trades((1001, 3.0, 1), (1006, 0.1, 1)), book(1001, 2, 1006, 1, 1005), "pessimistic")
    assert r.filled == pytest.approx(0.01) and r.how == "through"


def test_book_crossing_our_quote_fills():
    # the best ask came down to our bid price without a trade: that seller would have hit us
    o = Order(side=BID, price=1000, qty=0.01, Q=3.0)
    r = update_order(o, *trades(), book(1000, 3.0, 999, 2.0, 1000), "pessimistic")
    assert r.filled == pytest.approx(0.01) and r.how == "cross"


# ------------------------------------------------------------------ cancellations and the size cap
def test_cancellations_ordering_of_rules():
    """Level 1000: 10 BTC at the start, we have 6 ahead. 1 BTC trades, the level ends at 4 (min 4):
    unexplained decrease C = 10 - 4 - 1 = 5."""
    bk = book(best_prev=1000, size_prev=10.0, best_end=1000, size_min=4.0, opp_end=1001)
    res = {}
    for rule in ("pessimistic", "proportional", "optimistic"):
        o = Order(side=BID, price=1000, qty=0.01, Q=6.0)
        update_order(o, *trades((1000, 1.0, -1)), bk, rule)
        res[rule] = o.Q
    # after the trade Q = 5, level after trades = 9
    assert res["optimistic"] == pytest.approx(0.0)                 # 5 cancels ahead first: 5 - 5
    assert res["proportional"] == pytest.approx(5 - 5 * 5 / 9)     # 2.222
    assert res["pessimistic"] == pytest.approx(4.0)                # no cancels ahead, but capped by the size (4)
    assert res["optimistic"] <= res["proportional"] <= res["pessimistic"]


def test_cap_queue_ahead_at_min_displayed_size():
    o = Order(side=BID, price=1000, qty=0.01, Q=8.0)
    update_order(o, *trades(), book(1000, 9.0, 1000, 2.0, 1001), "pessimistic")
    assert o.Q == 2.0           # the level showed only 2 BTC at some point: at most 2 can be ahead


def test_unseen_level_learns_queue_when_it_becomes_best():
    o = Order(side=BID, price=998, qty=0.01, Q=INF)      # joined 2 ticks behind the touch
    update_order(o, *trades((998, 5.0, -1)), book(1000, 4.0, 1000, 3.0, 1001), "pessimistic")
    assert o.Q == INF          # still behind: prints at our price do not reach an unknown queue
    update_order(o, *trades(), book(1000, 3.0, 998, 7.0, 999), "pessimistic")
    assert o.Q == 7.0          # level is now best: we are behind all 7 BTC shown


def test_level_back_after_leaving_keeps_place():
    o = Order(side=ASK, price=1001, qty=0.01, Q=1.5)
    # a better ask (1000) appears: our level is now behind, its queue is unobserved and unchanged
    update_order(o, *trades(), book(1001, 4.0, 1000, 0.5, 999), "optimistic")
    assert o.Q == 1.5
    # the better ask leaves; our level is best again showing 2.5: we keep 1.5 ahead (never cancelled)
    update_order(o, *trades(), book(1000, 0.5, 1001, 2.5, 1000), "optimistic")
    assert o.Q == 1.5


def test_our_level_emptied_without_trades_puts_us_first():
    o = Order(side=BID, price=1000, qty=0.01, Q=3.0)
    update_order(o, *trades(), book(1000, 3.0, 999, 6.0, 1001), "pessimistic")
    assert o.Q == 0.0 and o.qty == 0.01     # everyone at 1000 cancelled; we would be the best bid alone
    r = update_order(o, *trades((999, 0.002, -1)), book(999, 6.0, 999, 6.0, 1001), "pessimistic")
    assert r.filled == pytest.approx(0.01)   # a seller hitting 999 would have hit our better bid first
