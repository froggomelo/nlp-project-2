"""Floor on stricter translated pairs (post-hoc, no retraining, CPU)

Recomputes the floor and gap on kept test pairs above higher similarity thresholds, from the saved test
predictions, so the classifier and language removal (from the 0.55 training pairs) stay fixed.
Exploratory: chosen after seeing test results, so no threshold is selected from them.

Usage:
    python -m src.posthoc.floor_checks > results/floor_checks.txt
"""
import numpy as np
import pandas as pd

THRESHOLDS, B = [0.55, 0.60, 0.65], 1000
rng = np.random.default_rng(13)
p = pd.read_parquet("results/raw/stage1_test_predictions.parquet")
pred = p.set_index("chunk_id")


def ba(q):
    return 0.5 * ((q[q.y == 1].pred == 1).mean() + (q[q.y == 0].pred == 0).mean())


all_ba = ba(p)
cand = pd.read_parquet("results/raw/pair_candidates.parquet")
cand = cand[(cand["split"] == "test") & (cand["sim"] >= THRESHOLDS[0])]
groups = [g for _, g in p.groupby("issue")]

print(f"all test chunks BA={all_ba:.3f}\n")
print("Test pairs above stricter thresholds (language removal still from the 0.55 training pairs)")
for t in THRESHOLDS:
    k = cand[cand.sim >= t]
    ids = set(k.xh_id) | set(k.en_id)
    f, floors, gaps = ba(pred.loc[list(ids)]), [], []
    for _ in range(B):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        q = s[s.chunk_id.isin(ids)]
        if q.y.nunique() == 2:
            floors.append(ba(q))
            gaps.append(ba(s) - floors[-1])
    fci, gci = np.percentile(floors, [2.5, 97.5]), np.percentile(gaps, [2.5, 97.5])
    print(f"   sim >= {t:.2f}: {len(k)} pairs floor={f:.3f} [{fci[0]:.3f}, {fci[1]:.3f}] "
          f"gap={all_ba - f:.3f} [{gci[0]:.3f}, {gci[1]:.3f}]")
