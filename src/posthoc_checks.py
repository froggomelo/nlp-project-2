"""Four post-hoc checks on the Stage 1 test predictions (no retraining, CPU, ~3 min).

1. floor vs pair similarity     2. gap without number-heavy chunks     3. gap per year
4. gap per period (1884-89, 1890-94), each against its own floor: yearly floors rest on too few pairs

Usage:
    python src/posthoc_checks.py . > results/posthoc_checks.txt
"""
import re
import sys

import numpy as np
import pandas as pd

root = sys.argv[1]
B, rng = 1000, np.random.default_rng(13)
DIGIT = re.compile(r"\d")


def ba(df):
    return 0.5 * ((df.loc[df.y == 1, "pred"] == 1).mean() + (df.loc[df.y == 0, "pred"] == 0).mean())


def boot(df, fn):
    issues = df["issue"].unique()
    groups = {i: g for i, g in df.groupby("issue")}
    vals = []
    for _ in range(B):
        s = pd.concat([groups[i] for i in rng.choice(issues, len(issues))])
        try:
            vals.append(fn(s))
        except ZeroDivisionError:
            pass
    vals = np.array([v for v in vals if np.isfinite(v)])
    return np.percentile(vals, [2.5, 97.5])


def numeric_share(t):
    toks = t.split()
    return sum(bool(DIGIT.search(w)) for w in toks) / max(len(toks), 1)


for run, thr in [("raw", 0.55), ("raw__multilingual-e5-base", 0.8571373618568346)]:
    p = pd.read_parquet(f"{root}/results/{run}/stage1_test_predictions.parquet")
    cand = pd.read_parquet(f"{root}/results/{run}/pair_candidates.parquet")
    pairs = cand[(cand.split == "test") & (cand.sim >= thr)]
    print(f"\n######## {run}: {len(p)} test chunks, {p.in_pair.sum()} in pairs, {len(pairs)} test pairs")
    gap = lambda d: ba(d) - ba(d[d.in_pair])
    print(f"check: all={ba(p):.3f} floor={ba(p[p.in_pair]):.3f} gap={gap(p):.3f}")

    # 1. floor by pair-similarity bin (a pair = its two chunks)
    print("\n1. floor by pair similarity")
    sim_of = pd.concat([pairs.set_index("xh_id")["sim"], pairs.set_index("en_id")["sim"]])
    sim_of = sim_of[~sim_of.index.duplicated()]
    pp = p[p.in_pair].copy()
    pp["sim"] = pp.chunk_id.map(sim_of)
    print(f"   matched sims for {pp.sim.notna().sum()} of {len(pp)} pair chunks")
    pp["bin"] = pd.qcut(pp.sim, 4)
    for b, g in pp.groupby("bin", observed=True):
        lo, hi = boot(g, ba)
        print(f"   sim {str(b):18s} n={len(g):4d}  floor={ba(g):.3f} [{lo:.3f}, {hi:.3f}]")
    r = np.corrcoef(pp.sim, (pp.pred == pp.y).astype(float))[0, 1]
    print(f"   correlation(sim, classifier correct on pair chunk) = {r:+.3f}")

    # 2. gap without number-heavy / table chunks (floor kept from all pairs, and from filtered pairs)
    print("\n2. gap without number-heavy chunks")
    p["num"] = p.text.map(numeric_share)
    p["table"] = p.text.str.contains(r"\||\+-{3,}")
    for cut in [0.3, 0.2, 0.1, 0.05]:
        keep = p[(p.num < cut) & ~p.table]
        lo, hi = boot(keep, gap)
        sh_xh = 1 - (keep.y == 1).sum() / (p.y == 1).sum()
        sh_en = 1 - (keep.y == 0).sum() / (p.y == 0).sum()
        print(f"   digits<{cut:.0%} & no table: dropped xh {sh_xh:.1%} en {sh_en:.1%} | "
              f"all={ba(keep):.3f} floor={ba(keep[keep.in_pair]):.3f} gap={gap(keep):.3f} [{lo:.3f}, {hi:.3f}]")

    # 3. per year
    print("\n3. per year (gap uses that year's pairs; pooled-floor gap uses all test pairs)")
    floor_all = ba(p[p.in_pair])
    for yr, g in p.groupby("year"):
        npair = g.in_pair.sum()
        fl = ba(g[g.in_pair]) if npair >= 20 else np.nan
        lo, hi = boot(g, ba)
        print(f"   {yr}: issues={g.issue.nunique():2d} chunks={len(g):5d} pairchunks={npair:4d} "
              f"all={ba(g):.3f} [{lo:.3f},{hi:.3f}] floor={fl:.3f} gap={ba(g)-fl:.3f} pooled-floor gap={ba(g)-floor_all:.3f}")
    yrs = p.groupby("year")[["y", "pred"]].apply(ba)
    slope = np.polyfit(yrs.index, yrs.values, 1)[0]
    sl = []
    issues = p.issue.unique()
    groups = {i: g for i, g in p.groupby("issue")}
    for _ in range(300):
        s = pd.concat([groups[i] for i in rng.choice(issues, len(issues))])
        y2 = s.groupby("year")[["y", "pred"]].apply(ba)
        sl.append(np.polyfit(y2.index, y2.values, 1)[0])
    print(f"   trend in all-chunk BA: {slope*10:+.3f} per decade, 95% CI [{np.percentile(sl,2.5)*10:+.3f}, {np.percentile(sl,97.5)*10:+.3f}]")

    # 4. two periods, each with its own floor; paired bootstrap over issues for the change
    print("\n4. per period, each against its own floor")
    p["period"] = np.where(p.year <= 1889, "1884-89", "1890-94")
    res = {}
    for per, g in p.groupby("period"):
        lo, hi = boot(g, gap)
        res[per] = gap(g)
        print(f"   {per}: issues={g.issue.nunique():2d} pair chunks={g.in_pair.sum():4d} all={ba(g):.3f} "
              f"floor={ba(g[g.in_pair]):.3f} gap={gap(g):.3f} [{lo:.3f}, {hi:.3f}]")
    groups = {i: g for i, g in p.groupby("issue")}
    late = {i: g.period.iloc[0] == "1890-94" for i, g in groups.items()}
    e_issues = [i for i in groups if not late[i]]
    l_issues = [i for i in groups if late[i]]
    diffs = []
    for _ in range(B):
        e = pd.concat([groups[i] for i in rng.choice(e_issues, len(e_issues))])
        l_ = pd.concat([groups[i] for i in rng.choice(l_issues, len(l_issues))])
        diffs.append(gap(l_) - gap(e))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"   change in gap (1890-94 minus 1884-89): {res['1890-94'] - res['1884-89']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
