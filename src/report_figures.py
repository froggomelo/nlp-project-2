"""Figures for the report, plotted from saved results (no retraining; CPU, ~10 s).

  report/figures/fig_results.pdf   main figure: synthetic check, and floor vs all-chunk accuracy by
                                   spelling and period
  report/figures/fig_topics.pdf    supplementary: isiXhosa share of every topic cluster

Reads results/synthetic/synthetic.json, results/compare/compare.json,
results/raw/stage1_test_predictions.parquet and results/raw/qualitative.json.

Usage:
    python src/report_figures.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("report/figures")
BLUE, VERMILION, GREEN = "#0072B2", "#D55E00", "#009E73"  # Okabe-Ito, checked with the dataviz palette validator
GREY, INK, MUTED, GRID = "#8c8b86", "#111111", "#52514e", "#e6e6e6"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": 0.9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 8, "axes.titlepad": 5, "axes.labelsize": 7.5,
                     "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
                     "xtick.major.size": 3, "ytick.major.size": 3,
                     "legend.frameon": False, "legend.fontsize": 6.8, "figure.dpi": 200})


def ba(d):
    return 0.5 * ((d.loc[d.y == 1, "pred"] == 1).mean() + (d.loc[d.y == 0, "pred"] == 0).mean())


def light_grid(ax):
    ax.grid(color=GRID, lw=0.7)


def synthetic(ax):
    syn = pd.DataFrame(json.loads(Path("results/synthetic/synthetic.json").read_text()))
    lines = [("oracle", 0.0, "-", INK, r"True $\Delta$"), ("oracle", 0.4, "--", GREY, r"True $\Delta$, 40% false"),
             ("pairs", 0.0, "-", BLUE, "Pair removal"), ("pairs", 0.4, "--", VERMILION, "Pair removal, 40% false")]
    dodge = {"oracle": -0.005, "pairs": 0.005}  # small x offset so the true value and the method don't hide each other
    for method, ff, ls, colour, label in lines:
        g = syn[syn.false_frac == ff].groupby("delta")[f"{method}_gap"].agg(["mean", "std"])
        g.index = g.index + dodge[method]
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=colour, alpha=0.12, lw=0)
        ax.plot(g.index, g["mean"], color=colour, ls=ls, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.5, label=label)
        if method == "pairs":  # label the method's own end point
            ax.text(g.index[-1] + 0.012, g["mean"].iloc[-1], f"{g['mean'].iloc[-1]:.3f}".lstrip("0"), va="center", color=INK)
    g = syn[syn.false_frac == 0.0].groupby("delta")["mean-centre_gap"].mean()
    ax.plot(g.index, g.values, color=GREEN, ls=":", marker="s", ms=3, mec="white", mew=0.5, lw=1.3, label="Mean-centring")
    ax.legend(loc="upper left", ncol=2, labelspacing=0.2, handlelength=2.0, columnspacing=1.2, borderaxespad=0.1)
    ax.set_xlabel(r"Planted difference $\delta$")
    ax.set_ylabel(r"$\Delta$ on test chunks")
    ax.set_xticks([0, 0.1, 0.2, 0.4])
    ax.set_xlim(-0.02, 0.46)
    ax.set_ylim(-0.012, 0.175)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    light_grid(ax)
    ax.set_title("(a) Recovering a planted difference")


def floor_dumbbells(ax):
    cmp = {r["variant"]: r for r in json.loads(Path("results/compare/compare.json").read_text())["table"]}
    p = pd.read_parquet("results/raw/stage1_test_predictions.parquet")
    early, late = p[p.year <= 1889], p[p.year >= 1890]
    rows = [  # label, all-chunk accuracy, floor
        ("Raw", cmp["raw"]["all"], cmp["raw"]["floor"]),
        ("Modern", cmp["modern"]["all"], cmp["modern"]["floor"]),
        ("Aspiration only", cmp["modern_asp"]["all"], cmp["modern_asp"]["floor"]),
        ("Prefix only", cmp["modern_prefix"]["all"], cmp["modern_prefix"]["floor"]),
        (None, None, None),
        ("1884–89", ba(early), ba(early[early.in_pair])),
        ("1890–94", ba(late), ba(late[late.in_pair])),
    ]
    labels, ys = [], []
    for i, (label, a, f) in enumerate(rows):
        y = len(rows) - i
        if label is None:
            ax.axhline(y, color=GRID, lw=0.7)
            continue
        ax.plot([f, a], [y, y], color=BLUE, alpha=0.5, lw=1.8, solid_capstyle="butt", zorder=2)
        ax.plot([a], [y], "o", ms=5, color=BLUE, mec="white", mew=0.5, zorder=3)
        ax.plot([f], [y], "o", ms=5, mfc="white", mec=BLUE, mew=1.2, zorder=3)
        ax.text(a + 0.012, y, f"{a - f:.3f}".lstrip("0"), va="center", color=INK)
        labels.append(label)
        ys.append(y)
    ax.plot([], [], "o", ms=5, mfc="white", mec=BLUE, mew=1.2, ls="none", label="Floor")
    ax.plot([], [], "o", ms=5, color=BLUE, mec="white", mew=0.5, ls="none", label="All test chunks")
    ax.legend(loc="upper right", handlelength=1.0, labelspacing=0.3, borderaxespad=0.1)
    ax.axvline(0.5, color=INK, lw=0.9, ls=":", zorder=1)
    ax.text(0.507, len(rows) + 0.25, "Chance", va="center", color=INK)
    ax.set_yticks(ys, labels)
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.set_xlim(0.48, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Balanced accuracy (BA)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_title(r"(b) $\Delta$ across spellings and periods")


def main_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.45), gridspec_kw=dict(width_ratios=[1.0, 1.1]))
    synthetic(axes[0])
    floor_dumbbells(axes[1])
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "fig_results.pdf")


TOPIC_LABELS = {5: "market prices", 16: "tables", 20: "land sales, surveys", 23: "scores, name lists",
                19: "meeting and show reports", 12: "sport", 17: "association meetings", 10: "Transvaal news",
                14: "women, marriage", 29: "land rates", 13: "health, medicine", 8: "religion",
                15: "schools", 4: "Kimberley, Hofmeyr", 1: "Cape politicians", 30: "crops, Basutoland",
                18: "legislation, land", 25: "liquor licensing", 22: "councils, commissions",
                27: "letters to the editor", 9: "questions and replies", 28: "land, rivers, frontier",
                3: "names (cluster 3)", 21: "names (cluster 21)", 24: "names (cluster 24)", 31: "names (cluster 31)"}


def topics_figure():
    q = json.loads(Path("results/raw/qualitative.json").read_text())
    c = pd.DataFrame(q["clusters"])
    c = c[c.n > 1].sort_values("xh_share")
    base = (c.n * c.xh_share).sum() / c.n.sum()
    fig, ax = plt.subplots(figsize=(4.8, 5.6))
    above = c.xh_share >= base
    y = np.arange(len(c)) + above.to_numpy() * 1.2  # a gap between the two groups for their labels
    ax.barh(y, c.xh_share, color=[BLUE if a else VERMILION for a in above], height=0.66, edgecolor="white", lw=0.6)
    ax.axvline(base, color=INK, lw=0.8, ls="--", zorder=3)
    ax.text(base, y.max() + 0.6, f" All chunks {base:.2f}", color=MUTED, va="bottom")
    # name the two groups beside their bars, so the colours need no legend
    top_above, top_below = y[above.to_numpy()].max(), y[~above.to_numpy()].max()
    for top, text, colour in [(top_above, "More isiXhosa than all chunks", BLUE),
                              (top_below, "Less isiXhosa than all chunks", VERMILION)]:
        ax.plot([0.015], [top + 0.85], "s", ms=4, color=colour, clip_on=False)
        ax.text(0.035, top + 0.85, text, fontsize=6, color=MUTED, va="center")
    names = [TOPIC_LABELS.get(k, "cluster " + str(k)) for k in c.cluster]
    ax.set_yticks(y, [f"{s[0].upper() + s[1:]} ({n:,})" for s, n in zip(names, c.n)])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, y.max() + 1.5)
    ax.set_xlabel("Share of the topic's chunks that are isiXhosa")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_topics.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main_figure()
    topics_figure()
    print("wrote", OUT / "fig_results.pdf", OUT / "fig_topics.pdf")
