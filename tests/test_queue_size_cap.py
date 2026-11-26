"""Stage 5: the optional size cap of the queue model (``update_order(..., size_cap=True)``).

Stage 4's 0.01 BTC order is filled in full by any trade through its price. A 0.5 BTC execution order is
not: had it been in the book, the aggressor would have eaten the queue ahead of it first and then only as
much of our order as the aggressor's own size allowed. With size_cap=True:

* trade through: fill = min(qty, V_at_or_through - Q_ahead), where V_at_or_through is the volume the hitting
  aggressors traded at our price or worse (for them); an unseen queue (Q = inf) is taken to be the whole
  volume traded at our price (all of it ahead of us).
* book crossing our price: fill = min(qty, displayed opposite size at the end of the bucket), a lower bound
  on the size of the order that crossed.
Without size_cap (the default), stage 4 behaviour is unchanged.
"""

import math

import numpy as np
import pytest

from mms.replay.queue import ASK, BID, BucketBook, Order, update_order


def trades(*rows):
    p, q, s = zip(*rows)
    return np.array(p, dtype=np.int64), np.array(q, float), np.array(s, dtype=np.int8)


def test_through_fill_is_capped_by_aggressor_volume_beyond_queue():
    # we bid 0.5 BTC at 1000 with 0.3 ahead; sellers trade 0.4 at 1000 and 0.2 at 999 (through us)
    tr = trades((1000, 0.4, -1), (999, 0.2, -1))
    bk = BucketBook(best_prev=1000, size_prev=0.3, best_end=999, size_min=1.0, opp_end=1001, opp_size_end=2.0)
    o = Order(side=BID, price=1000, qty=0.5, Q=0.3)
    r = update_order(o, *tr, bk, "proportional", size_cap=True)
    # V = 0.6 at or through, 0.3 ahead -> 0.3 for us; the level was swept, so the rest is alone at 1000
    assert r.how == "through" and r.filled == pytest.approx(0.3) and o.qty == pytest.approx(0.2)
    assert o.Q == 0.0
    # same, but the ask ends at 1000 on our remaining bid: crossing, capped by its displayed size 2.0 -> all
    # filled (the bucket's fill is reported under its first cause, "through")
    bk2 = BucketBook(best_prev=1000, size_prev=0.3, best_end=999, size_min=1.0, opp_end=1000, opp_size_end=2.0)
    o2 = Order(side=BID, price=1000, qty=0.5, Q=0.3)
    r2 = update_order(o2, *tr, bk2, "proportional", size_cap=True)
    assert r2.how == "through" and r2.filled == pytest.approx(0.5) and o2.qty == 0.0


def test_through_fill_unseen_queue_counts_volume_at_our_price_as_ahead():
    # our ask at 1003 behind the touch (Q unknown); buyers lift 1.0 at 1003 and 0.15 at 1004
    bk = BucketBook(best_prev=1001, size_prev=1.0, best_end=1005, size_min=1.0, opp_end=1002, opp_size_end=1.0)
    o = Order(side=ASK, price=1003, qty=0.5, Q=math.inf)
    r = update_order(o, *trades((1003, 1.0, 1), (1004, 0.15, 1)), bk, "proportional", size_cap=True)
    assert r.filled == pytest.approx(0.15)


def test_uncapped_default_is_stage4_full_fill():
    bk = BucketBook(best_prev=1000, size_prev=0.3, best_end=999, size_min=1.0, opp_end=1001, opp_size_end=0.01)
    o = Order(side=BID, price=1000, qty=0.5, Q=0.3)
    r = update_order(o, *trades((1000, 0.4, -1), (999, 0.2, -1)), bk, "proportional")
    assert r.how == "through" and r.filled == pytest.approx(0.5)


def test_cross_fill_capped_by_opposite_size():
    # our bid at 1000, no trades; the ask comes down to 1000 with 0.2 displayed
    bk = BucketBook(best_prev=1000, size_prev=3.0, best_end=999, size_min=1.0, opp_end=1000, opp_size_end=0.2)
    o = Order(side=BID, price=1000, qty=0.5, Q=2.0)
    r = update_order(o, np.array([], np.int64), np.array([]), np.array([], np.int8), bk, "proportional",
                     size_cap=True)
    assert r.how == "cross" and r.filled == pytest.approx(0.2) and o.qty == pytest.approx(0.3)
    o3 = Order(side=BID, price=1000, qty=0.5, Q=2.0)
    r3 = update_order(o3, np.array([], np.int64), np.array([]), np.array([], np.int8), bk, "proportional")
    assert r3.filled == pytest.approx(0.5)      # stage 4: full


def test_small_order_same_with_and_without_cap():
    # a 0.01 BTC order and a big sweep: the cap does not bind
    bk = BucketBook(best_prev=1000, size_prev=2.0, best_end=998, size_min=1.0, opp_end=999, opp_size_end=1.0)
    tr = trades((1000, 2.0, -1), (999, 1.0, -1), (998, 0.5, -1))
    a = Order(side=BID, price=1000, qty=0.01, Q=2.0)
    b = Order(side=BID, price=1000, qty=0.01, Q=2.0)
    ra = update_order(a, *tr, bk, "proportional")
    rb = update_order(b, *tr, bk, "proportional", size_cap=True)
    assert ra.filled == rb.filled == pytest.approx(0.01)
