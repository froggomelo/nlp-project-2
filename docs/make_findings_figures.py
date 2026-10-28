"""Figures and numbers for docs/findings.pdf (run from the project root; CPU, ~3 min).

    python docs/make_findings_figures.py
"""
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
B, rng = 1000, np.random.default_rng(13)
RUNS = {"LaBSE": ("raw", 0.55), "E5": ("raw__multilingual-e5-base", 0.8571373618568346)}
COL = {"LaBSE": "#2a78d6", "E5": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
DIGIT = re.compile(r"\d")
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "figure.dpi": 200})


def ba(df):
    return 0.5 * ((df.loc[df.y == 1, "pred"] == 1).mean() + (df.loc[df.y == 0, "pred"] == 0).mean())


def boot(df, fn, n=B):
    groups = {i: g for i, g in df.groupby("issue")}
    keys = list(groups)
    vals = [fn(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(n)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return np.percentile(vals, [2.5, 97.5])


def gap(d):
    return ba(d) - ba(d[d.in_pair])


nums = {}
data = {}
for enc, (run, thr) in RUNS.items():
    p = pd.read_parquet(f"results/{run}/stage1_test_predictions.parquet")
    cand = pd.read_parquet(f"results/{run}/pair_candidates.parquet")
    pairs = cand[(cand.split == "test") & (cand.sim >= thr)]
    sim_of = pd.concat([pairs.set_index("xh_id")["sim"], pairs.set_index("en_id")["sim"]])
    p["sim"] = p.chunk_id.map(sim_of[~sim_of.index.duplicated()])
    p["num"] = p.text.map(lambda t: sum(bool(DIGIT.search(w)) for w in t.split()) / max(len(t.split()), 1))
    p["table"] = p.text.str.contains(r"\||\+-{3,}")
    data[enc] = p

# 1. floor by similarity quartile, and the gap measured against the most similar half of pairs
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for j, (enc, p) in enumerate(data.items()):
    pp = p[p.in_pair].copy()
    pp["q"] = pd.qcut(pp.sim, 4, labels=False)
    rows = []
    for q, g in pp.groupby("q"):
        lo, hi = boot(g, ba)
        rows.append((q, ba(g), lo, hi))
    r = np.array(rows)
    x = r[:, 0] + (j - 0.5) * 0.12
    ax.errorbar(x, r[:, 1], yerr=[r[:, 1] - r[:, 2], r[:, 3] - r[:, 1]], color=COL[enc], marker="o", ms=4,
                capsize=0, elinewidth=1, label=enc)
    top = pp[pp.q >= 2].issue.unique()
    top_ids = set(pp[pp.q >= 2].chunk_id)
    p_top = p.assign(in_pair=p.chunk_id.isin(top_ids))
    lo, hi = boot(p_top, gap)
    nums[f"{enc}_floor_by_quartile"] = [round(v, 3) for v in r[:, 1]]
    nums[f"{enc}_gap_top_half"] = [round(gap(p_top), 3), round(lo, 3), round(hi, 3)]
    nums[f"{enc}_floor_top_half"] = round(ba(p_top[p_top.in_pair]), 3)
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.text(-0.25, 0.505, "chance", color=MUTED, ha="left", va="bottom", fontsize=7)
ax.set_xticks(range(4), ["least\nsimilar", "", "", "most\nsimilar"])
ax.set_xlabel("translated pairs, by similarity quarter")
ax.set_ylabel("floor (balanced accuracy)")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig_floor_by_similarity.png")

# 2. gap after removing number-heavy and table chunks
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for enc, p in data.items():
    pts = []
    for cut in [1.01, 0.3, 0.2, 0.1, 0.05]:
        keep = p[(p.num < cut) & ~p.table] if cut <= 1 else p
        dropped = 1 - (keep.y == 1).sum() / (p.y == 1).sum()
        lo, hi = boot(keep, gap, n=500)
        pts.append((dropped * 100, gap(keep), lo, hi))
    a = np.array(pts)
    ax.errorbar(a[:, 0], a[:, 1], yerr=[a[:, 1] - a[:, 2], a[:, 3] - a[:, 1]], color=COL[enc], marker="o",
                ms=4, elinewidth=1, label=enc)
    nums[f"{enc}_gap_by_removed_share"] = [[round(v, 3) for v in row] for row in a]
ax.set_ylim(0, 0.35)
ax.set_xlabel("isiXhosa chunks removed (%), number-heavy first")
ax.set_ylabel("content gap $\\Delta$")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig_gap_without_numbers.png")

# 3. all-chunk balanced accuracy by year
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for j, (enc, p) in enumerate(data.items()):
    rows = []
    for yr, g in p.groupby("year"):
        lo, hi = boot(g, ba, n=500)
        rows.append((yr, ba(g), lo, hi))
    a = np.array(rows)
    x = a[:, 0] + (j - 0.5) * 0.15
    ax.errorbar(x, a[:, 1], yerr=[a[:, 1] - a[:, 2], a[:, 3] - a[:, 1]], color=COL[enc], marker="o", ms=4,
                elinewidth=1, label=enc)
    nums[f"{enc}_all_by_year"] = {int(r[0]): round(r[1], 3) for r in a}
ax.set_xlabel("year")
ax.set_ylabel("balanced accuracy, all chunks")
ax.set_xticks(range(1884, 1895, 2))
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig_by_year.png")

# 4. modernisation: all vs floor per variant (LaBSE) - the mechanism of the gap change
cmp = json.loads(Path("results/compare/compare.json").read_text())["table"]
cmp_e5 = json.loads(Path("results/compare__multilingual-e5-base/compare.json").read_text())["table"]
rows = [(f"LaBSE {r['variant']}", r["all"], r["floor"], "LaBSE") for r in cmp] + \
       [(f"E5 {r['variant']}", r["all"], r["floor"], "E5") for r in cmp_e5]
names = {"raw": "raw", "modern": "modern", "modern_asp": "aspiration only", "modern_prefix": "prefix only"}
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for i, (lab, a_, f_, enc) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([f_, a_], [y, y], color=COL[enc], lw=2, solid_capstyle="round")
    ax.plot([f_], [y], "o", color="white", mec=COL[enc], ms=5, mew=1.5)
    ax.plot([a_], [y], "o", color=COL[enc], ms=5)
ax.set_yticks(range(len(rows)), [f"{lab.split()[0]}: {names[lab.split()[1]]}" for lab, *_ in rows][::-1])
ax.set_xlabel("balanced accuracy\n(open = floor, filled = all chunks)")
ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
fig.tight_layout()
fig.savefig(OUT / "fig_modernisation_mechanism.png")

Path("docs/figures/findings_numbers.json").write_text(json.dumps(nums, indent=1))
print(json.dumps(nums, indent=1))
