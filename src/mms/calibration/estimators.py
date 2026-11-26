r"""Estimators for the A-S model parameters sigma, A and k from Binance data.

Units throughout: time in seconds, prices in USDT (per 1 BTC).

1. Volatility sigma (USDT per sqrt(second))
-------------------------------------------
Model the mid as arithmetic Brownian motion, dS = sigma dW. Sampled every
dt seconds, the increments dS_i = S_{i+1} - S_i are i.i.d. N(0, sigma^2 dt), so

    E[dS_i^2] = sigma^2 dt   =>   sigma_hat^2 = (1 / (n dt)) * sum_i dS_i^2.

This is the maximum-likelihood estimator when the drift is known to be zero
(no mean is subtracted: over seconds the drift is negligible and estimating it
only adds noise). Increments touching a missing (NaN) mid are dropped, never
set to zero. ``step`` subsamples (every step-th point, dt -> step * dt), which is
how we check that 1-second sampling is not inflated by microstructure noise
(bid-ask bounce would make the 1 s estimate larger than the 60 s one).

2. Fill intensity lambda(delta) = A exp(-k delta): the touch estimator
---------------------------------------------------------------------
A resting ask at mid_0 + delta would have been (at least partly) filled during
a window [t_0, t_0 + Dt) if a buyer-initiated trade printed at a price
>= mid_0 + delta in that window (and symmetrically for the bid with
seller-initiated trades at <= mid_0 - delta). For each window we record the
*excursion*

    ask_exc = max(buy trade prices in the window) - mid_0   (-inf if no buys)
    bid_exc = mid_0 - min(sell trade prices in the window)  (-inf if no sells)

where mid_0 is the mid in force just before the window opens. For a grid of
distances delta, the fraction of windows with excursion >= delta estimates

    p(delta) = P(a fill at distance delta within Dt).

If fills at distance delta arrive as a Poisson process with rate lambda(delta),
the number N of arrivals in Dt is Poisson(lambda Dt), so

    p = P(N >= 1) = 1 - exp(-lambda Dt)   =>   lambda = -ln(1 - p) / Dt.

(For small p this is ~ p / Dt; the log corrects for several arrivals in one
window counting only once.) p = 1 gives lambda = infinity: such points carry no
information about lambda and are excluded from the fit (and counted).

Taking logs of the model, ln lambda(delta) = ln A - k delta, a straight line in
delta. Ordinary least squares of y_j = ln lambda_j on x_j = delta_j gives

    slope b = sum (x_j - x_bar)(y_j - y_bar) / sum (x_j - x_bar)^2,
    intercept a = y_bar - b x_bar,        k_hat = -b,  A_hat = exp(a),

and R^2 = 1 - SS_res / SS_tot measures how close to a straight line the points
are. Residuals that bend systematically (e.g. concave) mean the exponential
shape is wrong, whatever R^2 says.

3. Conditioning on volatility (added after the out-of-sample check, see DECISIONS.md)
------------------------------------------------------------------------------------
A and k fitted day by day move strongly with that day's sigma: when the mid is
more volatile, prices print further from the starting mid more often, so A
(the rate one tick beyond the touch, extrapolated to delta = 0) rises and the
decay k falls. A power law captures this with two numbers per parameter:

    A(sigma) = a_A * sigma^{b_A},      k(sigma) = a_k * sigma^{b_k}.

Taking logs, ln A = ln a_A + b_A ln sigma is again a straight line, fitted by
the same OLS formulas across days (x = ln sigma_day, y = ln A_day). sigma is
measured from the mid alone, so a live market maker can estimate it without
any information about its own fills.

4. Uncertainty across days
--------------------------
Days are the independent units (windows within a day are correlated through
the day's volatility). The bootstrap resamples *days* with replacement, pools
their hit counts, refits, and takes the 2.5% / 97.5% quantiles of A and k.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EPS = 1e-6  # price tolerance (USDT): 0.1-tick prices minus mids are not exact in floating point


def realised_sigma(mid: np.ndarray, dt_s: float, step: int = 1) -> float:
    """sigma_hat = sqrt( sum dS^2 / (n * dt) ), using every ``step``-th point (dt -> step * dt_s)."""
    x = np.asarray(mid, dtype=float)[::step]
    d = np.diff(x)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        raise ValueError("no valid increments")
    return float(np.sqrt(np.sum(d**2) / (len(d) * dt_s * step)))


def touch_excursions(snap: pd.DataFrame, trades: pd.DataFrame, window_s: int) -> pd.DataFrame:
    """Per window of ``window_s`` seconds: start time, mid_0, ask_exc, bid_exc (see module docstring).

    snap: 1-second snapshots (time_s, mid). trades: time [ms], price, is_buyer_maker.
    Windows start at snap.time_s[0] + j * window_s; windows whose mid_0 is NaN are dropped.
    """
    t0 = int(snap["time_s"].iloc[0])
    starts = snap[(snap["time_s"] - t0) % window_s == 0][["time_s", "mid"]].reset_index(drop=True)
    n_windows = len(starts)
    w_id = (trades["time"].to_numpy(np.int64) // 1000 - t0) // window_s
    price = trades["price"].to_numpy(float)
    buy = ~trades["is_buyer_maker"].to_numpy(bool)  # buyer was the aggressor: lifts asks
    ok = (w_id >= 0) & (w_id < n_windows)

    max_buy = np.full(n_windows, -np.inf)
    min_sell = np.full(n_windows, np.inf)
    np.maximum.at(max_buy, w_id[ok & buy], price[ok & buy])
    np.minimum.at(min_sell, w_id[ok & ~buy], price[ok & ~buy])

    mid0 = starts["mid"].to_numpy(float)
    out = pd.DataFrame({"time_s": starts["time_s"], "mid0": mid0,
                        "ask_exc": max_buy - mid0, "bid_exc": mid0 - min_sell})
    keep = np.isfinite(mid0)
    if (~keep).any():
        log.info("touch: dropped %d window(s) with no mid at the start", int((~keep).sum()))
    return out[keep].reset_index(drop=True)


def touch_curve(exc: np.ndarray, deltas: np.ndarray) -> tuple[np.ndarray, int]:
    """Number of windows with excursion >= delta (inclusive, to within EPS) for each delta, and n windows."""
    exc = np.sort(np.asarray(exc, dtype=float))
    n = len(exc)
    hits = n - np.searchsorted(exc, np.asarray(deltas, float) - EPS, side="left")
    return hits.astype(np.int64), n


def intensity_from_fraction(p, window_s: float):
    """lambda = -ln(1 - p) / Dt (infinite when p = 1)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return -np.log1p(-p) / window_s


def fit_exponential(deltas: np.ndarray, lam: np.ndarray) -> dict:
    """OLS fit of ln lambda = ln A - k delta. Returns A, k, r2, residuals (in ln lambda)."""
    x = np.asarray(deltas, dtype=float)
    lam = np.asarray(lam, dtype=float)
    if np.any(~np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("intensities must be finite and positive to take logs")
    if len(x) < 2:
        raise ValueError("need at least two points")
    y = np.log(lam)
    xb, yb = x.mean(), y.mean()
    slope = np.sum((x - xb) * (y - yb)) / np.sum((x - xb) ** 2)
    intercept = yb - slope * xb
    resid = y - (intercept + slope * x)
    ss_tot = np.sum((y - yb) ** 2)
    r2 = 1 - np.sum(resid**2) / ss_tot if ss_tot > 0 else 1.0
    return {"A": float(np.exp(intercept)), "k": float(-slope), "r2": float(r2), "residuals": resid}


def fit_mask(hits: np.ndarray, n: int, min_hits: int) -> np.ndarray:
    """Points used in a fit: at least ``min_hits`` hits (enough data) and p < 1 (finite lambda)."""
    hits = np.asarray(hits)
    return (hits >= min_hits) & (hits < n)


def fit_counts(hits: np.ndarray, n: int, deltas: np.ndarray, window_s: float, min_hits: int = 30) -> dict:
    """Touch counts -> intensities -> exponential fit over the usable deltas."""
    m = fit_mask(hits, n, min_hits)
    if m.sum() < 2:
        raise ValueError(f"only {m.sum()} usable delta points (min_hits={min_hits})")
    lam = intensity_from_fraction(hits[m] / n, window_s)
    fit = fit_exponential(np.asarray(deltas)[m], lam)
    fit.update(mask=m, lam=intensity_from_fraction(hits / n, window_s), n_used=int(m.sum()))
    return fit


def bootstrap_days(days: dict, deltas: np.ndarray, window_s: float, n_boot: int = 2000,
                   seed: int = 0, min_hits: int = 30) -> dict:
    """Pooled fit over all days plus a percentile bootstrap over days.

    ``days`` maps a day label to (hits array over deltas, number of windows).
    Returns {"A": (est, lo, hi), "k": (est, lo, hi), "r2": est, "boot": array of (A, k)}.
    The fit range (mask) is fixed from the full pooled data so every resample fits the same deltas.
    """
    keys = list(days)
    H = np.array([days[d][0] for d in keys], dtype=float)
    N = np.array([days[d][1] for d in keys], dtype=float)
    full = fit_counts(H.sum(0), N.sum(), deltas, window_s, min_hits)
    mask = full["mask"]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        h, n = H[idx].sum(0), N[idx].sum()
        lam = intensity_from_fraction(h[mask] / n, window_s)
        if np.any(~np.isfinite(lam)) or np.any(lam <= 0):
            continue  # cannot happen when every day has hits in range; counted below
        f = fit_exponential(np.asarray(deltas)[mask], lam)
        boots.append((f["A"], f["k"]))
    boots = np.array(boots)
    if len(boots) < n_boot:
        log.warning("bootstrap: %d of %d resamples unusable", n_boot - len(boots), n_boot)
    qA = np.quantile(boots[:, 0], [0.025, 0.975])
    qk = np.quantile(boots[:, 1], [0.025, 0.975])
    return {"A": (full["A"], *qA), "k": (full["k"], *qk), "r2": full["r2"], "boot": boots, "fit": full}


def fit_loglog(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS of ln y on ln x: y = scale * x^slope. Returns slope, scale, r2 and a predict(x) function."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("log-log fit needs positive values")
    # Reuse the linear fit: ln y = ln(scale) + slope * ln x  <=>  "lambda" = y, "delta" = ln x, k = -slope
    f = fit_exponential(np.log(x), y)
    slope, scale = -f["k"], f["A"]
    return {"slope": slope, "scale": scale, "r2": f["r2"], "predict": lambda v: scale * np.asarray(v, float) ** slope}
