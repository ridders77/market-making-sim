r"""Adverse selection measured from the data: markouts, order-flow imbalance (OFI) and its predictive power.

1. Markouts ("how much does the price move against a market maker right after they're filled")
-------------------------------------------------------------------------------------------
Every aggTrade has a passive side: ``is_buyer_maker`` True means the resting order was a *bid*
(the maker bought), False means the maker sold. With m_0 the mid in force just before the trade
and m_h the mid h seconds later, the move *against* the maker is

    adverse_h = -s (m_h - m_0),   s = +1 if the maker bought, -1 if the maker sold.

Positive = the maker lost money on the position (bought and the price fell, or sold and it rose).
delta = |price - m_0| is how far from the mid the maker's order sat. m_h comes from the 1-second
snapshots: the first snapshot second >= (trade time + h), i.e. the book as of between h and h + 1 s
after the trade (1-second resolution). Past the end of the day: NaN.

2. Order-flow imbalance
-----------------------
With B_j, S_j the aggressive buy / sell volume (BTC) printed during second j,

    OFI_N(s) = sum_{j = s-N}^{s-1} (B_j - S_j),        imbalance_N(s) = OFI_N(s) / sum (B_j + S_j),

using only trades before second s. The imbalance is in [-1, 1] and does not depend on how busy
the market is; it is NaN when no volume traded (never set to 0).

3. Predictive power
-------------------
Target: the next-h-second mid move in bps, y = 1e4 (m_{s+h} - m_s) / m_s. In-sample, OLS
y = a + b x gives the slope b and R^2. Out of sample the calibration (a, b) is applied unchanged:

    R^2_oos = 1 - sum (y - a - b x)^2 / sum y^2    (benchmark: predict no move, the martingale),
    hit rate = share of seconds where sign(a + b x) = sign(y), among y != 0.

4. From the measured slope to the simulator's signal
----------------------------------------------------
In the simulator, x is an OU signal with time constant tau and the mid drifts at kappa x, so
E[move over h | x] = kappa tau (1 - e^{-h/tau}) x = beta(h) x. Measuring beta(h) at several h and
fitting this curve gives (kappa, tau): for each tau on a grid, kappa has a closed form
(least squares through the origin: kappa = sum beta_h g_h / sum g_h^2, g_h = tau (1 - e^{-h/tau}));
we keep the tau with the smallest squared error.

5. Fill tilt
------------
In the simulator lambda_ask = lambda(delta) e^{theta x - theta^2/2}, lambda_bid = lambda(delta) e^{-theta x - theta^2/2},
so ln(lambda_ask / lambda_bid) = 2 theta x. Bin the windows by x, compute the ask and bid fill
shares in each bin, and regress the log ratio on the bin's mean x: slope = 2 theta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def markouts(trades: pd.DataFrame, snap: pd.DataFrame, horizons=(1, 5, 30, 60)) -> pd.DataFrame:
    """Per trade: delta (USDT), mid_before, maker_bought, adverse_h (USDT) and adverse_bps_h for each h."""
    t0 = int(snap["time_s"].iloc[0])
    mid = snap["mid"].to_numpy(float)
    if not np.array_equal(snap["time_s"].to_numpy() - t0, np.arange(len(mid))):
        raise ValueError("snapshots must be on a gap-free 1-second grid")
    m0 = ((trades["bid"] + trades["ask"]) / 2).to_numpy(float)
    s = np.where(trades["is_buyer_maker"].to_numpy(bool), 1.0, -1.0)
    out = pd.DataFrame({"delta": np.abs(trades["price"].to_numpy(float) - m0), "mid_before": m0,
                        "maker_bought": s > 0})
    t_s = trades["time"].to_numpy(np.int64) / 1000.0
    for h in horizons:
        idx = np.ceil(t_s + h - 1e-9).astype(np.int64) - t0
        ok = (idx >= 0) & (idx < len(mid))
        mh = np.full(len(trades), np.nan)
        mh[ok] = mid[idx[ok]]
        adv = -s * (mh - m0)
        out[f"adverse_{h}"] = adv
        out[f"adverse_bps_{h}"] = 1e4 * adv / m0
    return out


def rolling_ofi(buy: np.ndarray, sell: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """OFI_n(s) and imbalance_n(s) from per-second volumes, using seconds s-n .. s-1 (NaN for s < n)."""
    buy, sell = np.asarray(buy, float), np.asarray(sell, float)
    cn = np.concatenate([[0.0], np.cumsum(buy - sell)])
    ct = np.concatenate([[0.0], np.cumsum(buy + sell)])
    m = len(buy)
    ofi, vol = np.full(m, np.nan), np.full(m, np.nan)
    s = np.arange(n, m)
    ofi[s] = cn[s] - cn[s - n]
    vol[s] = ct[s] - ct[s - n]
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(vol > 1e-12, ofi / vol, np.nan)
    return ofi, imb


def forward_move(mid: np.ndarray, h: int) -> np.ndarray:
    """1e4 (m_{s+h} - m_s) / m_s in bps; NaN where either mid is missing or s + h is past the end."""
    mid = np.asarray(mid, float)
    y = np.full(len(mid), np.nan)
    y[:-h] = 1e4 * (mid[h:] - mid[:-h]) / mid[:-h]
    return y


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS y = a + b x: slope b = cov(x, y)/var(x), intercept a = mean y - b mean x, R^2 = 1 - var(resid)/var(y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xb, yb = x.mean(), y.mean()
    b = np.sum((x - xb) * (y - yb)) / np.sum((x - xb) ** 2)
    a = yb - b * xb
    resid = y - a - b * x
    return {"slope": float(b), "intercept": float(a), "r2": float(1 - np.sum(resid**2) / np.sum((y - yb) ** 2)),
            "n": len(x)}


def prediction_stats(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> dict:
    """Out-of-sample R^2 vs the zero forecast and hit rate (sign agreement where y != 0) for a fixed line."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    yhat = intercept + slope * x
    nz = (y != 0) & (yhat != 0)
    return {"r2_oos": float(1 - np.sum((y - yhat) ** 2) / np.sum(y**2)),
            "hit_rate": float(np.mean(np.sign(yhat[nz]) == np.sign(y[nz]))), "n": len(y), "n_hit": int(nz.sum())}


def fit_drift_decay(h: np.ndarray, beta: np.ndarray, taus: np.ndarray | None = None) -> dict:
    """Fit beta(h) = kappa tau (1 - exp(-h/tau)) by a grid over tau and closed-form kappa (section 4)."""
    h, beta = np.asarray(h, float), np.asarray(beta, float)
    taus = np.exp(np.linspace(np.log(0.2), np.log(600), 4000)) if taus is None else taus
    best = None
    for tau in taus:
        g = tau * (1 - np.exp(-h / tau))
        kappa = np.sum(beta * g) / np.sum(g * g)
        sse = np.sum((beta - kappa * g) ** 2)
        if best is None or sse < best[2]:
            best = (tau, kappa, sse)
    tau, kappa, sse = best
    return {"tau": float(tau), "kappa": float(kappa), "sse": float(sse),
            "fitted": kappa * tau * (1 - np.exp(-h / tau))}


def fit_tilt(x: np.ndarray, ask_hit: np.ndarray, bid_hit: np.ndarray, n_bins: int = 10) -> dict:
    """theta from ln(ask fill share / bid fill share) = 2 theta x, by OLS over quantile bins of x (section 5)."""
    x = np.asarray(x, float)
    ask_hit, bid_hit = np.asarray(ask_hit, bool), np.asarray(bid_hit, bool)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    xm, lr = [], []
    for j in range(n_bins):
        m = b == j
        pa, pb = ask_hit[m].mean(), bid_hit[m].mean()
        if m.sum() == 0 or pa == 0 or pb == 0:
            continue
        xm.append(x[m].mean())
        lr.append(np.log(pa / pb))
    f = ols(np.array(xm), np.array(lr))
    return {"theta": f["slope"] / 2, "intercept": f["intercept"], "r2": f["r2"], "bin_x": xm, "bin_log_ratio": lr}
