"""Where does removing the language signal work, and where does it fail? (post-hoc, no retraining, CPU)

For each kept test pair (raw, LaBSE) the saved Stage 1 predictions say whether the classifier still
tells the two halves apart. A pair is
  blind      both halves get the same predicted section (language removal worked on this pair)
  separated  the isiXhosa half is predicted isiXhosa and the English half English (it did not)
  reversed   the opposite (the content pulls each half towards the other section)
The pairs are then compared on features that could explain the outcome, and examples are printed.
Uses the saved embeddings and language basis only to measure how much of each chunk lay in the
removed directions.

Usage:
    python -m src.posthoc.pair_cases . > results/pair_cases.md
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

root = sys.argv[1]
THR = 0.55
WORD = re.compile(r"[A-Za-z]+")
DIGIT = re.compile(r"\d")

lex = set(pd.read_parquet(f"{root}/data/modern/lexicon.parquet")["word"])
cand = pd.read_parquet(f"{root}/results/raw/pair_candidates.parquet")
pairs = cand[(cand.split == "test") & (cand.sim >= THR)].reset_index(drop=True)
pred = {v: pd.read_parquet(f"{root}/results/{v}/stage1_test_predictions.parquet").set_index("chunk_id")
        for v in ["raw", "modern"]}

# share of each embedding's norm that lay in the removed language directions
ids = open(f"{root}/data/embeddings/raw_ids.txt", encoding="utf-8").read().split()
row = {c: i for i, c in enumerate(ids)}
E = np.load(f"{root}/data/embeddings/raw_LaBSE.npy", mmap_mode="r")
V = np.load(f"{root}/results/raw/language_basis.npy")


def lang_share(cid):
    x = np.asarray(E[row[cid]], dtype=float)
    return float(np.linalg.norm(V @ x) ** 2 / (x @ x))


def oov(t):
    w = [x.lower() for x in WORD.findall(t)]
    return sum(x not in lex for x in w) / max(1, len(w))


def digits(t):
    w = t.split()
    return sum(bool(DIGIT.search(x)) for x in w) / max(1, len(w))


def caps(t):
    w = WORD.findall(t)
    return sum(x[0].isupper() for x in w) / max(1, len(w))


def outcome(px, pe):
    if px == pe:
        return "blind"
    return "separated" if px == 1 else "reversed"


for v in ["raw", "modern"]:
    pairs[f"out_{v}"] = [outcome(pred[v].at[a, "pred"], pred[v].at[b, "pred"]) for a, b in zip(pairs.xh_id, pairs.en_id)]
pairs["p_xh_xh"] = pairs.xh_id.map(pred["raw"].p_xh)
pairs["p_xh_en"] = pairs.en_id.map(pred["raw"].p_xh)
pairs["spread"] = pairs.p_xh_xh - pairs.p_xh_en          # >0: pushed apart in the right direction
pairs["page"] = pairs.xh_id.str.extract(r"_p(\d+)_")[0].astype(int)
pairs["oov_xh"] = pairs.xh_text.map(oov)
pairs["digits"] = (pairs.xh_text.map(digits) + pairs.en_text.map(digits)) / 2
pairs["caps_xh"] = pairs.xh_text.map(caps)
pairs["len_ratio"] = pairs.xh_text.str.split().str.len() / pairs.en_text.str.split().str.len()
pairs["lang_xh"] = pairs.xh_id.map(lang_share)
pairs["lang_en"] = pairs.en_id.map(lang_share)

F = ["sim", "margin", "n_shared", "oov_xh", "digits", "caps_xh", "len_ratio", "lang_xh", "lang_en", "page", "year"]
print(f"# Pair outcomes after language removal (raw, LaBSE, k={len(V)}, {len(pairs)} test pairs)\n")
print("Outcome counts (raw):", pairs.out_raw.value_counts().to_dict())
print("Raw -> modern:\n")
print(pd.crosstab(pairs.out_raw, pairs.out_modern).to_markdown())
print("\n## Feature means by outcome\n")
print(pairs.groupby("out_raw")[F].mean().round(3).to_markdown())
print("\n## Spearman correlation with spread (p_xh of isiXhosa half minus p_xh of English half)\n")
for f in F:
    r, pv = spearmanr(pairs[f], pairs.spread)
    print(f"- {f}: rho={r:+.3f} (p={pv:.3g})")
print("\nOutcome by sim quartile:\n")
print(pd.crosstab(pd.qcut(pairs.sim, 4), pairs.out_raw, normalize="index").round(2).to_markdown())
print("\nOutcome by page of isiXhosa half:\n")
print(pd.crosstab(pairs.page, pairs.out_raw).to_markdown())


def show(df, title, n=4):
    print(f"\n## {title}\n")
    for _, r in df.head(n).iterrows():
        print(f"### {r.xh_id} / {r.en_id}  sim={r.sim:.3f} anchors={r.n_shared} ({r.shared})  "
              f"p_xh: xh={r.p_xh_xh:.2f} en={r.p_xh_en:.2f}  modern: {r.out_modern}  oov={r.oov_xh:.2f}\n")
        print(f"- xh: {r.xh_text[:400]}")
        print(f"- en: {r.en_text[:400]}\n")


s = pairs.sort_values("sim", ascending=False)
show(s[s.out_raw == "separated"].sort_values("spread", ascending=False), "Separated most strongly")
show(s[s.out_raw == "separated"], "Separated despite highest similarity")
show(s[s.out_raw == "blind"], "Blind, highest similarity")
show(s[s.out_raw == "blind"].sort_values("n_shared", ascending=False), "Blind, most shared anchors")
show(s[s.out_raw == "reversed"], "Reversed")
show(s[(s.out_raw == "separated") & (s.out_modern == "blind")], "Separated in raw, blind after modernising")
