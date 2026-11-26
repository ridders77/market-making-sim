r"""Queue model: what happens to one of our resting limit orders during one 100 ms bucket of real data.

We replay the recorded best bid/ask (with sizes) and trades, and insert a small virtual order. We assume
our 0.01 BTC order does not change anybody else's behaviour (a limitation, stated in the report). Prices
are integer ticks (1 tick = 0.1 USDT) so that "same price" is an exact comparison.

State of an order: side, price p, remaining quantity, and Q = the volume (BTC) resting *ahead of us* at p.
Q = inf means "unknown": we joined a level behind the touch, whose size the best-bid/ask feed does not show.

Joining (done by the environment): a new order goes to the back of the queue, so Q = the displayed size at
p if p is the best price on its side; Q = 0 if p improves on the best price (we are alone at a new level);
Q = inf if p is behind the best price (unseen depth; the "back" convention, see ``update_order``).

One bucket, in this order (the order of events inside 100 ms is unknown; trades first is the conservative
choice: they are charged against the queue as it stood at the start of the bucket):

1. Trade through: an aggressor traded at a price *worse for them* than ours (a seller below our bid, a
   buyer above our ask). Every order at p must have been consumed first, so we are filled in full.
2. Traded volume at p, V (aggressors on the other side only), eats the queue ahead first, then us:
       Q <- Q - V;  if Q < 0: fill min(remaining, -Q), Q <- 0.
   So a fill happens exactly when the volume traded at our price exceeds the queue ahead.
3. Book crossing: the opposite best price reached p (an ask at or below our bid). In the real book that
   cannot happen while bids rest at p, so the level was empty and the incoming order would have matched us:
   filled in full.
4. Cancellations. If p was the best price at the start and at the end of the bucket, the displayed size
   fell by size_prev - size_min, of which V is explained by trades. The rest,
       C = max(0, size_prev - size_min - V),
   was cancelled. The data does not say whether those orders were ahead of us or behind us:
       pessimistic:  none ahead                       (ahead = 0)
       proportional: in proportion to the queue      (ahead = C * Q / level, level = size_prev - V)
       optimistic:   ahead first                     (ahead = min(C, Q))
   The three rules bracket the truth; every main result is reported under all three.
5. Size cap. Q can never exceed the displayed size at p (our order is virtual, so the displayed size is
   exactly the real orders ahead of us plus those behind). Orders ahead only ever leave, so Q at the end
   of the bucket is at most the *smallest* size the level showed during the bucket: Q <- min(Q, size_min).
   This applies under every rule (even "pessimistic" must accept cancels ahead when the size forces it),
   and it is how an unseen queue (Q = inf) becomes known when its level becomes the best price.
6. If p is better than the real best price at the end (bid above the best bid: everyone at p cancelled
   without trading), the level is empty in the real book and we would be alone there: Q <- 0.
7. If p is behind the best price, its queue is not observed and Q is unchanged. When the level becomes
   best again we keep our place (only capped by step 5): an order that was never cancelled keeps its
   time priority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BID, ASK = 1, -1          # our side: +1 we buy (bid), -1 we sell (ask)
RULES = ("pessimistic", "proportional", "optimistic")


@dataclass
class Order:
    side: int             # BID or ASK
    price: int            # ticks
    qty: float            # remaining quantity (BTC)
    Q: float              # queue ahead (BTC); math.inf = unknown (level behind the touch)


@dataclass(frozen=True)
class BucketBook:
    """The book on our order's side over one bucket (prices in ticks, sizes in BTC)."""
    best_prev: int        # our side's best price at the start of the bucket
    size_prev: float      # its displayed size at the start
    best_end: int         # our side's best price at the end of the bucket
    size_min: float       # smallest displayed size at best_end during the bucket
    opp_end: int          # the opposite side's best price at the end of the bucket


@dataclass(frozen=True)
class BucketResult:
    filled: float         # BTC filled in this bucket
    how: str              # "", "queue", "through" or "cross"


def cancels_ahead(rule: str, C: float, Q: float, level: float) -> float:
    """Part of the cancelled volume C that was ahead of us (Q ahead out of ``level`` at the price)."""
    if rule == "pessimistic":
        return 0.0
    if rule == "proportional":
        share = 1.0 if level <= 0 else min(1.0, Q / level)
        return C * share
    if rule == "optimistic":
        return min(C, Q)
    raise ValueError(f"unknown cancel rule {rule!r}; use one of {RULES}")


def update_order(order: Order, tr_price: np.ndarray, tr_qty: np.ndarray, tr_side: np.ndarray,
                 bk: BucketBook, rule: str) -> BucketResult:
    """Apply one bucket (steps 1-7 of the module docstring) to ``order`` in place; return what filled.

    tr_price, tr_qty, tr_side: the bucket's trades (ticks, BTC, aggressor side +1 buy / -1 sell)."""
    s, p = order.side, order.price
    # aggressors that can hit us: sellers for our bid, buyers for our ask
    hits = tr_side == -s
    if hits.any():
        px = tr_price[hits]
        if (s == BID and px.min() < p) or (s == ASK and px.max() > p):
            return _fill_all(order, "through")
        V = float(tr_qty[hits][px == p].sum())
    else:
        V = 0.0
    filled = 0.0
    if V > 0 and math.isfinite(order.Q):
        order.Q -= V
        if order.Q < 0:
            filled = min(order.qty, -order.Q)
            order.qty -= filled
            order.Q = 0.0
    if order.qty <= 0:
        return BucketResult(filled, "queue")
    if (s == BID and bk.opp_end <= p) or (s == ASK and bk.opp_end >= p):
        rest = _fill_all(order, "cross")
        return BucketResult(filled + rest.filled, "cross")
    better_than_best = (s == BID and p > bk.best_end) or (s == ASK and p < bk.best_end)
    if better_than_best:
        order.Q = 0.0
    elif p == bk.best_end:
        if p == bk.best_prev and math.isfinite(order.Q):
            C = max(0.0, bk.size_prev - bk.size_min - V)
            if C > 0:
                order.Q -= cancels_ahead(rule, C, order.Q, bk.size_prev - V)
        order.Q = max(0.0, min(order.Q, bk.size_min))
    return BucketResult(filled, "queue" if filled > 0 else "")


def _fill_all(order: Order, how: str) -> BucketResult:
    f = order.qty
    order.qty = 0.0
    order.Q = 0.0
    return BucketResult(f, how)
