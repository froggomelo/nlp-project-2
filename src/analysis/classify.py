"""Stage 1b: can the content of a chunk predict which section (isiXhosa or English) it came from?

Trap 1 (the section IS the language) is handled in two ways:
  * Language removal: the language subspace is estimated ONLY from translated pairs in train
    issues (their content is the same, so their difference is language). The top-k directions of
    the pair differences are projected out of every embedding. Plain per-language mean-centring is
    kept as a comparison, because it also removes any genuine average content difference.
  * Translated-pair floor: balanced accuracy on held-out translated pairs measures whatever language
    or spelling signal survives. Only accuracy ABOVE that floor counts as a content difference.

k (number of removed directions) is chosen on validation as the smallest k whose pair floor is
<= --floor-target; C (logistic-regression strength) is tuned on validation for every k.
The final gap (all-chunk accuracy minus pair floor) is reported on test with a bootstrap CI over issues.

For the raw-vs-modern comparison, every variant uses the SAME reference pairs and the SAME k as the
raw run (--pairs-from raw --k-from raw), so a change in the floor reflects the text, not a re-tuned
language-removal step. C is still tuned per variant.

Non-linear check: a linear probe cannot use a difference whose class means are equal, which is
exactly what mean-centring produces. So the gap is also reported for a cosine kNN probe (trained on a
class-balanced sample of train chunks, number of neighbours tuned on validation) for no removal,
mean-centring and the chosen k.

Usage:
    python -m src.analysis.classify --variant raw [--threshold 0.55]
    python -m src.analysis.classify --variant modern --pairs-from raw --k-from raw
    python -m src.analysis.classify --variant raw --threshold 0.65 --run-name raw_thr0.65   # robustness run
"""
import argparse
import gc
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

from src.common import load_embeddings, load_usable, run_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K_GRID = [0, 1, 2, 4, 8, 16, 32, 64]
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
KNN_K = [5, 25, 101]  # odd, so majority votes cannot tie


def language_basis(emb, idx_xh, idx_en, k):
    """Top-k right singular vectors of the (uncentred) xh-minus-en pair differences."""
    if k == 0:
        return np.zeros((0, emb.shape[1]), dtype=np.float32)
    D = emb[idx_xh] - emb[idx_en]
    _, _, vt = np.linalg.svd(D, full_matrices=False)
    return vt[:k]


def project_out(X, V):
    return X - (X @ V.T) @ V if len(V) else X


def ba(model, X, y):
    return balanced_accuracy_score(y, model.predict(X))


def fit_best_c(Xtr, ytr, Xva, yva, seed):
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed).fit(Xtr, ytr)
        s = ba(m, Xva, yva)
        if best is None or s > best[0]:
            best = (s, C, m)
        # each fit leaves a float64 copy of the training data (~0.3 GB) in reference cycles; without an
        # explicit collection these pile up across the 54 fits and Colab kills the process
        del m
        gc.collect()
    return best


def knn_check(X, y, sets, pair_idx, seed, max_per_class=15000):
    """Gap for a cosine kNN probe: n_neighbours tuned on validation all-chunk BA, then test all/floor/gap."""
    rng = np.random.default_rng(seed)
    tr = sets["train"]
    n = min(max_per_class, *(int((y[tr] == c).sum()) for c in (0, 1)))
    tr = np.concatenate([rng.choice(tr[y[tr] == c], n, replace=False) for c in (0, 1)])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # euclidean on unit vectors = cosine
    nn = NearestNeighbors(n_neighbors=max(KNN_K)).fit(Xn[tr])
    res = {}
    for s in ["val", "test"]:
        votes = y[tr][nn.kneighbors(Xn[sets[s]], return_distance=False)]  # neighbour labels, nearest first
        in_pair = np.isin(sets[s], pair_idx[s])
        ys = y[sets[s]]
        for kk in KNN_K:
            pred = (votes[:, :kk].mean(1) > 0.5).astype(int)
            a = balanced_accuracy_score(ys, pred)
            f = balanced_accuracy_score(ys[in_pair], pred[in_pair])
            res[(s, kk)] = (a, f, a - f)
    best = max(KNN_K, key=lambda kk: res[("val", kk)][0])
    a, f, g = res[("test", best)]
    return dict(n_neighbors=best, n_train_per_class=n, val_all=res[("val", best)][0],
                val_floor=res[("val", best)][1], test_all=a, test_floor=f, test_gap=g)


def pair_rows(df, pairs, split):
    p = pairs[pairs["split"] == split]
    pos = pd.Series(df.index, index=df["chunk_id"])
    return pos[p["xh_id"]].to_numpy(), pos[p["en_id"]].to_numpy(), p["issue"].to_numpy()


def bootstrap_gap(y, pred, issue, pair_mask, n_boot, seed):
    """Resample test issues with replacement; recompute all-chunk BA, pair floor and gap."""
    rng = np.random.default_rng(seed)
    issues = np.unique(issue)
    by_issue = {i: np.flatnonzero(issue == i) for i in issues}
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([by_issue[i] for i in rng.choice(issues, len(issues))])
        pm = idx[pair_mask[idx]]
        if len(np.unique(y[pm])) < 2:
            continue
        a = balanced_accuracy_score(y[idx], pred[idx])
        f = balanced_accuracy_score(y[pm], pred[pm])
        out.append((a, f, a - f))
    out = np.array(out)
    lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
    return {"all": [float(lo[0]), float(hi[0])], "floor": [float(lo[1]), float(hi[1])],
            "gap": [float(lo[2]), float(hi[2])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--threshold", type=float, default=None,
                    help="pair similarity threshold; default: results/<variant>/pair_threshold.json")
    ap.add_argument("--floor-target", type=float, default=0.55)
    ap.add_argument("--model-tag", default="LaBSE")
    ap.add_argument("--pairs-from", default=None, help="variant whose pairs/threshold define the reference pairs")
    ap.add_argument("--k-from", default=None, help="variant whose chosen k is reused instead of re-selecting")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--run-name", default=None,
                    help="write to results/<run-name> instead of the variant's folder (robustness runs)")
    ap.add_argument("--no-knn", action="store_true", help="skip the kNN probe")
    args = ap.parse_args()
    res_dir = Path("results") / args.run_name if args.run_name else run_dir(args.variant, args.model_tag)
    pairs_dir = run_dir(args.pairs_from or args.variant, args.model_tag)
    chunks = "data/processed/chunks.parquet" if args.variant == "raw" else         f"data/processed/chunks_{args.variant}.parquet"
    df, emb = load_embeddings(load_usable(chunks), args.variant, args.model_tag)
    if set(df["split"]) != {"train", "val", "test"}:
        raise SystemExit("embeddings incomplete: wait for src/embedding/embed.py to finish")

    thr_file = pairs_dir / "pair_threshold.json"
    if args.threshold is not None:
        thr, thr_src = args.threshold, "command line"
    elif thr_file.exists():
        thr, thr_src = json.loads(thr_file.read_text())["threshold"], str(thr_file)
    else:
        raise SystemExit("no threshold: label pairs and run `pairs.py tune`, or pass --threshold")
    pairs = pd.read_parquet(pairs_dir / "pair_candidates.parquet")
    pairs = pairs[pairs["sim"] >= thr]
    n_pairs = pairs["split"].value_counts()
    if n_pairs.get("train", 0) < 20 or n_pairs.get("val", 0) < 5 or n_pairs.get("test", 0) < 5:
        raise SystemExit(f"too few reference pairs at threshold {thr} ({n_pairs.to_dict()}): "
                         "check the pair labels / threshold before running the classifier")

    y = (df["lang"] == "xh").to_numpy().astype(int)
    split = df["split"].to_numpy()
    tr_x, tr_e, _ = pair_rows(df, pairs, "train")
    sets = {s: np.flatnonzero(split == s) for s in ["train", "val", "test"]}
    pair_idx = {}
    for s in ["val", "test"]:
        px, pe, _ = pair_rows(df, pairs, s)
        pair_idx[s] = np.concatenate([px, pe])

    # Each method's transformed embeddings are built, scored on validation and test, then freed: keeping a
    # copy per method (9 x 68k x 768 floats) ran Colab out of memory. Only predictions are kept.
    full_basis = language_basis(emb, tr_x, tr_e, max(K_GRID))
    train_means = {lang: emb[((df["lang"] == lang).to_numpy()) & (split == "train")].mean(0) for lang in ["xh", "en"]}

    def transformed(k):
        """Embeddings with language removed: per-language mean-centring if k is None, else top-k pair directions."""
        if k is not None:
            return project_out(emb, full_basis[:k])
        X = emb.copy()
        for lang, mu in train_means.items():
            X[(df["lang"] == lang).to_numpy()] -= mu
        return X

    test_idx = sets["test"]
    pair_mask = np.zeros(len(df), bool)
    pair_mask[pair_idx["test"]] = True
    rows = []
    for name, k in [("mean-centre", None)] + [(f"pairs k={k}", k) for k in K_GRID]:
        X = transformed(k)
        s_val, C, model = fit_best_c(X[sets["train"]], y[sets["train"]], X[sets["val"]], y[sets["val"]], args.seed)
        floor_val = ba(model, X[pair_idx["val"]], y[pair_idx["val"]])
        pred_test = model.predict(X[test_idx])
        a = balanced_accuracy_score(y[test_idx], pred_test)
        f = balanced_accuracy_score(y[pair_mask], model.predict(X[pair_mask]))
        rows.append(dict(method=name, k=k, C=C, val_all=s_val, val_floor=floor_val, val_gap=s_val - floor_val,
                         test_all=a, test_floor=f, test_gap=a - f, _model=model, _pred_test=pred_test))
        del X
        gc.collect()
        print(f"{name:12s} C={C:<5} val all={s_val:.3f} floor={floor_val:.3f} gap={s_val - floor_val:+.3f}", flush=True)
    table = [{key: v for key, v in r.items() if not key.startswith("_")} for r in rows]

    pair_rows_k = [r for r in rows if r["k"] is not None]
    if args.k_from:
        k_fixed = json.loads((run_dir(args.k_from, args.model_tag) / "stage1.json").read_text())["chosen_k"]
        chosen = next(r for r in pair_rows_k if r["k"] == k_fixed)
    else:
        chosen = next((r for r in pair_rows_k if r["val_floor"] <= args.floor_target), pair_rows_k[-1])

    model = chosen["_model"]
    pred_all = np.full(len(df), -1)
    pred_all[test_idx] = chosen["_pred_test"]
    ci = bootstrap_gap(y[test_idx], pred_all[test_idx], df["issue"].to_numpy()[test_idx],
                       pair_mask[test_idx], args.n_boot, args.seed)
    c_edge = chosen["C"] in (min(C_GRID), max(C_GRID))
    if c_edge:
        print(f"WARNING: chosen C={chosen['C']} is at the edge of the grid {C_GRID}; consider widening it")
    degenerate = len(np.unique(pred_all[test_idx])) < 2
    if degenerate:
        print("WARNING: the chosen classifier predicts a single class on test; its gap and floor are meaningless")

    knn = {}
    if not args.no_knn:
        for r in rows:
            if r["method"] in ("mean-centre", "pairs k=0", chosen["method"]):
                knn[r["method"]] = knn_check(transformed(r["k"]), y, sets, pair_idx, args.seed)
                print(f"kNN {r['method']:12s} test all={knn[r['method']]['test_all']:.3f} "
                      f"floor={knn[r['method']]['test_floor']:.3f} gap={knn[r['method']]['test_gap']:+.3f}",
                      flush=True)

    res_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(variant=args.variant, threshold=thr, threshold_source=thr_src,
                   n_pairs={s: int((pairs["split"] == s).sum()) for s in ["train", "val", "test"]},
                   pairs_from=args.pairs_from or args.variant, k_from=args.k_from, model_tag=args.model_tag,
                   floor_target=args.floor_target, chosen=chosen["method"], chosen_k=chosen["k"],
                   chosen_C=chosen["C"], C_grid=C_GRID, C_at_grid_edge=c_edge, degenerate=degenerate,
                   test=next(t for t in table if t["method"] == chosen["method"]), test_ci=ci, table=table,
                   knn=knn)
    (res_dir / "stage1.json").write_text(json.dumps(summary, indent=1, default=float))

    # per-chunk test probabilities for the qualitative analysis (which isiXhosa content looks 'English-like')
    proba = model.predict_proba(transformed(chosen["k"])[test_idx])[:, 1]
    out = df.iloc[test_idx][["chunk_id", "issue", "year", "page", "lang", "text"]].assign(
        y=y[test_idx], pred=pred_all[test_idx], p_xh=proba, in_pair=pair_mask[test_idx])
    out.to_parquet(res_dir / "stage1_test_predictions.parquet", index=False)
    np.save(res_dir / "language_basis.npy", full_basis[:chosen["k"]])

    t = pd.DataFrame(table)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    tk = t[t["k"].notna()]
    xs = np.arange(len(tk))
    ax.plot(xs, tk["val_all"], "o-", label="all chunks")
    ax.plot(xs, tk["val_floor"], "s--", label="translated pairs (floor)")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xticks(xs, [str(int(k)) for k in tk["k"]])
    ax.set_xlabel("language directions removed (k)")
    ax.set_ylabel("validation balanced accuracy")
    ax.set_title(f"Section classifier, {args.variant} isiXhosa")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(res_dir / "fig_k_curve.png", dpi=200)

    print(f"\nchosen: {chosen['method']} (C={chosen['C']}); test all={summary['test']['test_all']:.3f} "
          f"floor={summary['test']['test_floor']:.3f} gap={summary['test']['test_gap']:+.3f} "
          f"95% CI {ci['gap'][0]:+.3f}..{ci['gap'][1]:+.3f}")


if __name__ == "__main__":
    main()
