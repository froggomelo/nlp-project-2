"""How much of the section signal is topic? (post-hoc, no classifier retraining, CPU)

Refits the topic clusters as qualitative.py does (raw, LaBSE, language removed, k=32, seed 13) and names
each refit cluster after the saved Figure S2 cluster whose distinctive words it shares most. Each
cluster's isiXhosa lean is its share of isiXhosa-section chunks among TRAIN chunks only. On test issues:
  1. a topic-only rule (section = isiXhosa if the chunk's topic leans isiXhosa) vs the saved classifier
  2. classifier error rate for chunks whose topic is typical vs atypical of their own section
  3. language left once topic is fixed: within-topic BA of the classifier
  4. per-topic table: accuracy on each section's chunks

Usage:
    python -m src.posthoc.topic_errors > results/topic_errors.txt
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import balanced_accuracy_score as bas

from src.analysis.classify import project_out
from src.common import load_embeddings, load_usable
from src.analysis.qualitative import distinctive_words
from src.figures.report_figures import TOPIC_LABELS

B, rng = 1000, np.random.default_rng(13)
q = json.load(open("results/raw/qualitative.json"))
df, emb = load_embeddings(load_usable("data/processed/chunks.parquet"), "raw", "LaBSE")
X = project_out(emb, np.load("results/raw/language_basis.npy"))
tr = df.index[df["split"] == "train"]
km = MiniBatchKMeans(q["k"], random_state=13, n_init=5, batch_size=4096).fit(X[tr])
df["cluster"] = km.predict(X)

# name refit clusters by distinctive-word overlap with the saved clusters
saved = {c["cluster"]: set(c["words_en"]) | set(c["words_xh"]) for c in q["clusters"]}
name = {}
for c, g in df.groupby("cluster"):
    rest = df[df.cluster != c]
    w = set(distinctive_words(g[g.lang == "en"].text, rest[rest.lang == "en"].text)) | \
        set(distinctive_words(g[g.lang == "xh"].text, rest[rest.lang == "xh"].text))
    best = max(saved, key=lambda s: len(w & saved[s]) / max(1, len(w | saved[s])))
    ov = len(w & saved[best]) / max(1, len(w | saved[best]))
    name[c] = f"{TOPIC_LABELS.get(best, 'cluster ' + str(best))}" if ov >= 0.2 else f"refit {c}"

base = (df.loc[tr, "lang"] == "xh").mean()
lean = df.loc[tr].groupby("cluster")["lang"].apply(lambda s: (s == "xh").mean())
p = pd.read_parquet("results/raw/stage1_test_predictions.parquet")
p["cluster"] = p.chunk_id.map(df.set_index("chunk_id").cluster)
p["lean"] = p.cluster.map(lean)
p["topic_pred"] = (p.lean > base).astype(int)


def boot(fn):
    groups = [g for _, g in p.groupby("issue")]
    v = [fn(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])


print(f"k={q['k']}, train isiXhosa share {base:.3f}, {len(p)} test chunks, {p.in_pair.sum()} in pairs\n")
print("1. Topic-only rule vs classifier (BA)")
for lab, col in [("topic only", "topic_pred"), ("classifier", "pred")]:
    a = bas(p.y, p[col]); f = bas(p[p.in_pair].y, p[p.in_pair][col])
    lo, hi = boot(lambda d: bas(d.y, d[col]))
    print(f"   {lab:10s} all={a:.3f} [{lo:.3f}, {hi:.3f}]  pairs={f:.3f}  gap={a - f:.3f}")
agree = (p.pred == p.topic_pred).mean()
print(f"   classifier agrees with the topic-only rule on {agree:.1%} of test chunks")

print("\n2. Classifier errors for chunks whose topic is typical vs atypical of their own section")
p["atypical"] = np.where(p.y == 1, p.lean < base, p.lean > base)
p["err"] = p.pred != p.y
for sec, g in p.groupby("y"):
    for at, h in g.groupby("atypical"):
        print(f"   {'isiXhosa' if sec else 'English '} section, topic {'atypical' if at else 'typical '}: "
              f"n={len(h):5d}  error rate={h.err.mean():.3f}")
share_err = p[p.err].atypical.mean()
print(f"   atypical-topic chunks are {p.atypical.mean():.1%} of test chunks but {share_err:.1%} of errors")
lo, hi = boot(lambda d: d[d.atypical].err.mean() - d[~d.atypical].err.mean())
print(f"   error rate atypical minus typical = {p[p.atypical].err.mean() - p[~p.atypical].err.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\n3. Language left once topic is fixed: classifier BA within each topic (topics with >=30 chunks per section)")
rows = []
for c, g in p.groupby("cluster"):
    nx, ne = (g.y == 1).sum(), (g.y == 0).sum()
    rows.append(dict(topic=name[c], lean=lean[c], n_xh=nx, n_en=ne,
                     acc_xh=(g[g.y == 1].pred == 1).mean() if nx else np.nan,
                     acc_en=(g[g.y == 0].pred == 0).mean() if ne else np.nan,
                     ba=bas(g.y, g.pred) if nx >= 30 and ne >= 30 else np.nan))
t = pd.DataFrame(rows).sort_values("lean")
w = t.dropna(subset=["ba"])
wavg = np.average(w.ba, weights=w.n_xh + w.n_en)
print(f"   {len(w)} topics; chunk-weighted mean within-topic BA = {wavg:.3f} (classifier overall {bas(p.y, p.pred):.3f})")

print("\n4. Per-topic table (sorted by isiXhosa lean)\n")
print(t.round(3).to_markdown(index=False))
