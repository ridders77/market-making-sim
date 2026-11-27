"""Matplotlib charts for strategy comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mms.sim.simulator import SimResult

COLOURS = {"avellaneda_stoikov": "#1f77b4", "symmetric": "#ff7f0e"}
LABELS = {"avellaneda_stoikov": "Avellaneda–Stoikov", "symmetric": "Symmetric (same avg spread)"}


def plot_sample_path(res: SimResult, path: int, title: str, out: Path,
                     ylabel_price: str = "price", xlabel_time: str = "time t") -> None:
    """Mid, bid/ask quotes and inventory over time for one path."""
    p = res.paths
    t, tq = p["t"][:-1], p["t"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1])
    ax1.plot(tq, p["mid"][path], color="black", lw=1.2, label="mid")
    ax1.plot(t, p["ask"][path], color="#d62728", lw=0.8, label="ask")
    ax1.plot(t, p["bid"][path], color="#2ca02c", lw=0.8, label="bid")
    ax1.set_ylabel(ylabel_price)
    ax1.legend(loc="best", frameon=False)
    ax1.set_title(title)
    ax2.step(tq, p["q"][path], where="post", color="#1f77b4")
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("inventory q (units)")
    ax2.set_xlabel(xlabel_time)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_distributions(results: dict[float, dict[str, SimResult]], attr: str, xlabel: str,
                       title: str, out: Path, discrete: bool = False) -> None:
    """One panel per gamma, overlaid histograms of ``attr`` for each strategy."""
    gammas = list(results)
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 4))
    for ax, g in zip(np.atleast_1d(axes), gammas):
        vals = {name: getattr(r, attr) for name, r in results[g].items()}
        allv = np.concatenate(list(vals.values()))
        if discrete:
            bins = np.arange(allv.min() - 0.5, allv.max() + 1.5)
        else:
            bins = np.linspace(allv.min(), allv.max(), 50)
        for name, v in vals.items():
            ax.hist(v, bins=bins, alpha=0.55, color=COLOURS.get(name), label=LABELS.get(name, name))
        ax.set_title(f"γ = {g}")
        ax.set_xlabel(xlabel)
    np.atleast_1d(axes)[0].set_ylabel("number of simulations")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
