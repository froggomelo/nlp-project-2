"""Figures for the report, plotted from saved results (no retraining; CPU, ~10 s).

  report/figures/fig_results.pdf   main figure: synthetic check, and floor vs all-chunk accuracy by
                                   spelling and period
  report/figures/fig_topics.pdf    supplementary: isiXhosa share of every topic cluster

Reads results/synthetic/synthetic.json, results/*/stage1.json, results/compare/compare.json,
results/raw/stage1_test_predictions.parquet, results/raw/pair_candidates.parquet,
results/raw/qualitative.json and docs/figures/findings_numbers.json.

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
LABSE, E5, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8c8b86", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "figure.dpi": 200})


def stage1(run):
    return json.loads(Path(f"results/{run}/stage1.json").read_text())


def ba(d):
    return 0.5 * ((d.loc[d.y == 1, "pred"] == 1).mean() + (d.loc[d.y == 0, "pred"] == 0).mean())


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=7.5, fontweight="bold", pad=4)


def synthetic(ax):
    syn = pd.DataFrame(json.loads(Path("results/synthetic/synthetic.json").read_text()))
    for method, colour, label in [("oracle", INK, "oracle"), ("pairs", LABSE, "pair removal")]:
        for ff, ls in [(0.0, "-"), (0.4, "--")]:
            g = syn[syn.false_frac == ff].groupby("delta")[f"{method}_gap"].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], color=colour, ls=ls, marker="o", ms=2.5, lw=1.2,
                        elinewidth=0.7, capsize=0, label=label if ff == 0 else None)
    g = syn[syn.false_frac == 0.0].groupby("delta")["mean-centre_gap"].mean()
    ax.plot(g.index, g.values, color=GREY, marker="s", ms=2.5, lw=1.2, label="mean-centring, none")
    ax.plot([], [], color=MUTED, ls="--", lw=1.2, label="40% false")
    ax.set_xlabel(r"planted gap $\delta$")
    ax.set_ylabel(r"test gap $\Delta$")
    ax.set_xticks([0, 0.1, 0.2, 0.4])
    ax.set_ylim(-0.015, 0.13)
    ax.legend(loc="upper left", ncol=2, fontsize=5.5, handlelength=2.0, borderaxespad=0.1, labelspacing=0.25,
              columnspacing=1.0)
    panel_label(ax, "(a) The method recovers known gaps")


def gap_forest(ax):
    raw, e5, strict = stage1("raw"), stage1("raw__multilingual-e5-base"), stage1("raw_thr0.65")
    filt = json.loads(Path("docs/figures/findings_numbers.json").read_text())["LaBSE_gap_by_removed_share"][-1]
    mc = next(r for r in raw["table"] if r["method"] == "mean-centre")["test_gap"]
    rows = [  # label, gap, CI or None, colour, filled
        (r"pair removal, $k$=64", raw["test"]["test_gap"], raw["test_ci"]["gap"], LABSE, True),
        ("kNN probe", raw["knn"]["pairs k=64"]["test_gap"], None, LABSE, False),
        ("E5 encoder", e5["test"]["test_gap"], e5["test_ci"]["gap"], E5, True),
        (r"pairs at $\geq$0.65", strict["test"]["test_gap"], strict["test_ci"]["gap"], LABSE, True),
        ("number-heavy chunks out", filt[1], filt[2:4], LABSE, True),
        ("mean-centring", mc, None, GREY, True),
    ]
    for i, (label, g, ci, colour, filled) in enumerate(rows):
        y = len(rows) - i
        if ci is not None:
            ax.plot(ci, [y, y], color=colour, lw=1.4, solid_capstyle="round")
        ax.plot([g], [y], "o" if label != "mean-centring" else "X", ms=4.5, color=colour,
                mfc=colour if filled else "white", mew=1.1)
    ax.set_yticks(list(range(len(rows), 0, -1)), [r[0] for r in rows])
    ax.axvline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlim(-0.02, 0.33)
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.set_xlabel(r"content gap $\Delta$ (95% CI)")
    ax.tick_params(axis="y", length=0)
    panel_label(ax, r"(b) The gap holds")


def floor_dumbbells(ax):
    cmp = {r["variant"]: r for r in json.loads(Path("results/compare/compare.json").read_text())["table"]}
    p = pd.read_parquet("results/raw/stage1_test_predictions.parquet")
    early, late = p[p.year <= 1889], p[p.year >= 1890]
    rows = [  # label, all-chunk accuracy, floor
        ("raw", cmp["raw"]["all"], cmp["raw"]["floor"]),
        ("modern", cmp["modern"]["all"], cmp["modern"]["floor"]),
        ("aspiration only", cmp["modern_asp"]["all"], cmp["modern_asp"]["floor"]),
        ("prefix only", cmp["modern_prefix"]["all"], cmp["modern_prefix"]["floor"]),
        (None, None, None),
        ("1884--89", ba(early), ba(early[early.in_pair])),
        ("1890--94", ba(late), ba(late[late.in_pair])),
    ]
    labels, ys = [], []
    for i, (label, a, f) in enumerate(rows):
        y = len(rows) - i
        if label is None:
            ax.axhline(y, color=MUTED, lw=0.4, alpha=0.5)
            continue
        ax.plot([f, a], [y, y], color=LABSE, lw=1.6, solid_capstyle="round")
        ax.plot([f], [y], "o", ms=4.5, mfc="white", mec=LABSE, mew=1.1)
        ax.plot([a], [y], "o", ms=4.5, color=LABSE)
        ax.text(a + 0.008, y, f"{a - f:.3f}".lstrip("0"), va="center", fontsize=6, color=MUTED)
        labels.append(label.replace("--", "–"))
        ys.append(y)
    ax.set_yticks(ys, labels)
    ax.plot([], [], "o", ms=4.5, mfc="white", mec=LABSE, mew=1.1, ls="none", label="floor")
    ax.plot([], [], "o", ms=4.5, color=LABSE, ls="none", label="all chunks")
    ax.legend(loc="lower left", fontsize=5.5, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=":")
    ax.set_xlim(0.49, 0.87)
    ax.set_xlabel("balanced accuracy")
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "(b) Spelling and time move the floor")


def main_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.34), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    synthetic(axes[0])
    floor_dumbbells(axes[1])
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_results.pdf")
    fig.savefig(OUT / "fig_results_preview.png", dpi=150)


TOPIC_LABELS = {5: "market prices", 16: "tables", 20: "land sales, surveys", 23: "scores, name lists",
                19: "meeting and show reports", 12: "sport", 17: "association meetings", 10: "Transvaal news",
                14: "women, marriage", 29: "land rates", 13: "health, medicine", 8: "religion",
                15: "schools", 4: "Kimberley, Hofmeyr", 1: "Cape politicians", 30: "crops, Basutoland",
                18: "legislation, land", 25: "liquor licensing", 22: "councils, commissions"}


def topics_figure():
    q = json.loads(Path("results/raw/qualitative.json").read_text())
    c = pd.DataFrame(q["clusters"])
    c = c[c.n > 1].sort_values("xh_share")
    base = (c.n * c.xh_share).sum() / c.n.sum()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    y = np.arange(len(c))
    ax.barh(y, c.xh_share, color=[LABSE if s >= base else E5 for s in c.xh_share], height=0.72)
    ax.axvline(base, color=INK, lw=0.8, ls="--")
    ax.text(base, len(c) - 0.2, f" corpus share {base:.2f}", fontsize=6.5, va="bottom")
    ax.set_yticks(y, [f"{TOPIC_LABELS.get(k, 'cluster ' + str(k))} ({n:,})" for k, n in zip(c.cluster, c.n)])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of the cluster's chunks that are isiXhosa")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_topics.pdf")
    fig.savefig(OUT / "fig_topics_preview.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main_figure()
    topics_figure()
    print("wrote", OUT / "fig_results.pdf", OUT / "fig_topics.pdf")
