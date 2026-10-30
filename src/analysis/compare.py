"""Stage 3: what does modernising the isiXhosa spelling change?

For each variant (raw, modern, and the rule ablations) this reads the Stage 1 test predictions and
the embeddings, then reports, relative to raw, with paired bootstrap CIs over test issues
(the same resampled issues for every variant):
  all     balanced accuracy of the section classifier on all test chunks
  floor   balanced accuracy on the reference translated pairs (language/spelling signal left over)
  gap     all - floor (the content difference between the two audiences)
  P@1     share of reference pairs whose English partner is the isiXhosa chunk's nearest English
          chunk in its issue (how well the encoder aligns the two sections)
  sim     mean cosine similarity of reference pairs
Plus the number of mutual-nearest-neighbour pairs above the raw threshold found in each variant.

Selection bias: reference pairs are mutual nearest neighbours in the BASE (raw) space, so P@1 and sim
on them favour the base by construction. The unbiased versions (suffix _lab) use the human-labelled
true pairs from the embedding-free 'anchor' source (label >= 1, source == anchor in
labels/pairs_labelled.csv), which no variant selected; report those for
alignment, and the reference pairs only for the (larger-sample) floor.

Reading the result (H2): if floor and all move together and the gap is unchanged, modernising only
removed spelling signal; if the gap moves, modernising changed the measured content difference.
"Unchanged" is an equivalence claim, so it is tested against a margin fixed in advance (--margin,
default 0.02 balanced accuracy) with two one-sided tests at alpha = 0.05, i.e. the 90% paired
bootstrap CI of the gap change must lie inside +/- margin:
  equivalent      90% CI inside +/- margin
  different       95% CI excludes 0 and the 90% CI is not inside the margin
  inconclusive    neither (more test issues would be needed to decide)

Usage:
    python -m src.analysis.compare --variants raw modern modern_asp modern_prefix modern_keepnames [--margin 0.02]
"""
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.common import load_embeddings, load_usable, run_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def chunks_path(variant):
    return "data/processed/chunks.parquet" if variant == "raw" else f"data/processed/chunks_{variant}.parquet"


def pair_metrics(variant, tag, ref):
    """Per reference pair: cosine similarity and whether the English partner is the top-1 match in its issue."""
    df, emb = load_embeddings(load_usable(chunks_path(variant)), variant, tag)
    pos = pd.Series(df.index, index=df["chunk_id"])
    xi, ei = pos[ref["xh_id"]].to_numpy(), pos[ref["en_id"]].to_numpy()
    sim = np.einsum("ij,ij->i", emb[xi], emb[ei])
    en_by_issue = {i: g.index[g["lang"] == "en"].to_numpy() for i, g in df.groupby("issue")}
    top1 = np.array([en_by_issue[iss][np.argmax(emb[en_by_issue[iss]] @ emb[x])] == e
                     for x, e, iss in zip(xi, ei, ref["issue"])])
    return sim, top1


def stats(pred_df, mask_issues=None):
    d = pred_df if mask_issues is None else pred_df[pred_df["issue"].isin(mask_issues)]
    a = balanced_accuracy_score(d["y"], d["pred"])
    p = d[d["in_pair"]]
    f = balanced_accuracy_score(p["y"], p["pred"]) if p["y"].nunique() == 2 else np.nan
    return a, f, a - f


def h2_verdict(ci90, ci95, margin):
    if -margin < ci90[0] and ci90[1] < margin:
        return "equivalent"
    if ci95[0] > 0 or ci95[1] < 0:
        return "different"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["raw", "modern", "modern_asp", "modern_prefix", "modern_keepnames"])
    ap.add_argument("--margin", type=float, default=0.02, help="equivalence margin for the gap change (H2)")
    ap.add_argument("--model-tag", default="LaBSE")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--labels", default="labels/pairs_labelled.csv")
    args = ap.parse_args()
    tag, base = args.model_tag, args.variants[0]

    s1 = {v: json.loads((run_dir(v, tag) / "stage1.json").read_text()) for v in args.variants}
    thr = s1[base]["threshold"]
    preds = {v: pd.read_parquet(run_dir(v, tag) / "stage1_test_predictions.parquet") for v in args.variants}
    ref = pd.read_parquet(run_dir(base, tag) / "pair_candidates.parquet")
    ref = ref[(ref["sim"] >= thr) & ref["split"].isin(["val", "test"])].reset_index(drop=True)
    pm = {v: pair_metrics(v, tag, ref) for v in args.variants}
    lab = None
    try:
        lab = pd.read_csv(args.labels)
        lab = lab[pd.to_numeric(lab["label"], errors="coerce") >= 1]
        if "source" in lab:  # only embedding-free anchor pairs are unbiased across variants
            lab = lab[lab["source"] == "anchor"]
        lab = lab.reset_index(drop=True)
        pm_lab = {v: pair_metrics(v, tag, lab) for v in args.variants}
        print(f"unbiased alignment metrics on {len(lab)} human-labelled true pairs")
    except (FileNotFoundError, KeyError):
        print("no labelled pairs yet: P@1/sim are on base-selected pairs and biased towards the base")

    rows = []
    for v in args.variants:
        a, f, g = stats(preds[v])
        cand = pd.read_parquet(run_dir(v, tag) / "pair_candidates.parquet")
        extra = dict(p_at_1_lab=pm_lab[v][1].mean(), sim_lab=pm_lab[v][0].mean()) if lab is not None else {}
        rows.append(dict(variant=v, all=a, floor=f, gap=g, p_at_1=pm[v][1].mean(), sim=pm[v][0].mean(), **extra,
                         pairs_found=int((cand["sim"] >= thr).sum()), chosen_k=s1[v]["chosen_k"],
                         C=s1[v]["chosen_C"]))
    table = pd.DataFrame(rows)

    # paired bootstrap over test issues (classifier) and over val+test issues (pair metrics)
    rng = np.random.default_rng(args.seed)
    test_issues = np.unique(preds[base]["issue"])
    ref_issues = np.unique(ref["issue"])
    ref_rows = {i: np.flatnonzero(ref["issue"].to_numpy() == i) for i in ref_issues}
    boot = {v: [] for v in args.variants}
    for _ in range(args.n_boot):
        ti = rng.choice(test_issues, len(test_issues))
        ri = np.concatenate([ref_rows[i] for i in rng.choice(ref_issues, len(ref_issues))])
        base_vals = None
        for v in args.variants:
            d = preds[v].set_index("issue").loc[ti].reset_index()
            a, f, g = stats(d)
            vals = np.array([a, f, g, pm[v][1][ri].mean(), pm[v][0][ri].mean()])
            if base_vals is None:
                base_vals = vals
            boot[v].append(vals - base_vals)
    names = ["all", "floor", "gap", "p_at_1", "sim"]
    for v in args.variants:
        b = np.array(boot[v])
        lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
        for j, n in enumerate(names):
            table.loc[table["variant"] == v, f"d_{n}"] = table.loc[table["variant"] == v, n].values[0] - \
                table.loc[table["variant"] == base, n].values[0]
            table.loc[table["variant"] == v, f"d_{n}_ci"] = f"[{lo[j]:+.3f}, {hi[j]:+.3f}]"
        lo90, hi90 = np.nanpercentile(b[:, 2], [5, 95])
        j = names.index("gap")
        table.loc[table["variant"] == v, "d_gap_ci90"] = f"[{lo90:+.3f}, {hi90:+.3f}]"
        if v == base:
            verdict = "--"
        elif s1[v].get("degenerate") or s1[base].get("degenerate"):
            verdict = "invalid (constant classifier)"
        else:
            verdict = h2_verdict((lo90, hi90), (lo[j], hi[j]), args.margin)
        table.loc[table["variant"] == v, "h2"] = verdict

    out = run_dir("compare", tag)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "compare.csv", index=False)
    (out / "compare.json").write_text(json.dumps(dict(base=base, threshold=thr, n_ref_pairs=len(ref),
                                                      margin=args.margin, table=table.to_dict("records")),
                                                 indent=1, default=float))
    with open(out / "compare.md", "w", encoding="utf-8") as fh:
        fh.write(f"Reference pairs: {len(ref)} (val+test, sim >= {thr} in '{base}'). "
                 f"H2 equivalence margin: +/-{args.margin} BA on the gap change (90% CI).\n\n")
        cols = ["variant", "all", "floor", "gap", "d_gap", "d_gap_ci", "d_gap_ci90", "h2", "d_floor",
                "d_floor_ci"] + (["p_at_1_lab", "sim_lab"] if lab is not None else []) + ["pairs_found"]
        fh.write(table[cols].to_markdown(index=False, floatfmt=".3f"))

    # central figure
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[3, 2]))
    x = np.arange(len(table))
    w = 0.26
    for j, (col, legend) in enumerate([("all", "all chunks"), ("floor", "translated-pair floor"),
                                    ("gap", "gap (content)")]):
        axes[0].bar(x + (j - 1) * w, table[col], w, label=legend)
    axes[0].axhline(0.5, color="grey", lw=0.7, ls=":")
    axes[0].set_xticks(x, table["variant"], fontsize=8)
    axes[0].set_ylabel("test balanced accuracy")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Section classifier", fontsize=9)
    p_col = "p_at_1_lab" if lab is not None else "p_at_1"
    axes[1].bar(x, table[p_col], 0.5, color="tab:purple")
    axes[1].set_xticks(x, table["variant"], fontsize=8)
    axes[1].set_ylabel("pair retrieval P@1" + ("" if lab is not None else " (biased, unlabelled)"))
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Cross-lingual alignment", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_compare.png", dpi=200)
    print(open(out / "compare.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
