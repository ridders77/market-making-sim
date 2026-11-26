"""Hand-built mini days for the execution tests (prices in ticks of 0.1 USDT)."""

import numpy as np

from mms.replay.data import ReplayDay


def make_day(bid, ask, bq, aq, trades=(), bq_min=None, aq_min=None) -> ReplayDay:
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
                     sell_min=sell_min, buy_max=buy_max, ofi_z=np.zeros(nb), sigma_hat=np.ones(nb),
                     typical_size=np.ones(nb))


N = 14


def flat_day(trades=(), bq=2.0, aq=3.0):
    return make_day([1000] * N, [1001] * N, [bq] * N, [aq] * N, trades=trades)
