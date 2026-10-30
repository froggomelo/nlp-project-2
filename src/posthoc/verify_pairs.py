"""Hand check of the kept test pairs that the floor is computed on (no retraining, CPU, seconds).

100 kept test pairs were sampled, 25 per similarity quartile, with the similarity hidden from labellers.
51 were labelled. A fluent isiXhosa speaker labelled 24 herself. She also checked all 27 that a team
member had labelled with AI translations and suggested labels, changing the ones she disagreed with.

data/pair_labels.csv holds one row per sampled pair:
    label     2 = translation, 1 = partial, 0 = different, blank = not labelled
    protocol  speaker_own, or ai_assisted_speaker_reviewed for the 27
    ai_label  the AI's suggestion, only compared with the final labels and never used as a label
    speaker_mark, speaker_mark_source  for the 27, her mark: written by her, or kept when she left the row blank

Usage:
    python -m src.posthoc.verify_pairs      # -> results/raw/pair_verification.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS = Path("data/pair_labels.csv")
OUT = Path("results/raw/pair_verification.json")
B = 1000


def ba(df):
    return 0.5 * ((df.loc[df.y == 1, "pred"] == 1).mean() + (df.loc[df.y == 0, "pred"] == 0).mean())


def boot(pred, fn, rng):
    issues = pred["issue"].unique()
    groups = {i: g for i, g in pred.groupby("issue")}
    vals = []
    for _ in range(B):
        v = fn(pd.concat([groups[i] for i in rng.choice(issues, len(issues))]))
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def shares(d):
    return {"n": int(len(d)), "translation": round(float((d.label == 2).mean()), 3),
            "partial": round(float((d.label == 1).mean()), 3), "different": round(float((d.label == 0).mean()), 3)}


def main():
    key = pd.read_csv(LABELS)
    done = key[key.label.notna()]
    print(f"labelled {len(done)} of {len(key)} pairs; by protocol {done.protocol.value_counts().to_dict()}")

    out = {"overall": shares(done), "by_quartile": {int(q): shares(g) for q, g in done.groupby("quartile")}}
    print("overall", out["overall"])
    for q, s in out["by_quartile"].items():
        print(f"  similarity quartile {q}: {s}")

    # AI suggestions are only compared with the final labels on the same rows
    both = done.dropna(subset=["ai_label"])
    out["ai_vs_human"] = {"ai_rows": int(key.ai_label.notna().sum()), "with_human_label": int(len(both)),
                          "raw": round(float((both.ai_label == both.label).mean()), 3),
                          "cohen_kappa": round(float(cohen_kappa_score(both.label, both.ai_label)), 3)}
    print("AI labels vs human (not used as labels):", out["ai_vs_human"])

    # floor and gap on verified pairs, from the saved test predictions (no retraining)
    pred = pd.read_parquet("results/raw/stage1_test_predictions.parquet")
    rng = np.random.default_rng(13)
    for name, keep in [("verified (label 2)", done.label == 2), ("label 1 or 2", done.label >= 1),
                       ("all sampled", done.label.notna())]:
        ids = set(done.loc[keep, "xh_id"]) | set(done.loc[keep, "en_id"])

        def floor(d):
            return ba(d[d.chunk_id.isin(ids)])

        def gap(d):
            return ba(d) - ba(d[d.chunk_id.isin(ids)])

        out[name] = {"pairs": int(keep.sum()), "floor": round(float(floor(pred)), 3),
                     "floor_ci": [round(v, 3) for v in boot(pred, floor, rng)],
                     "gap": round(float(gap(pred)), 3), "gap_ci": [round(v, 3) for v in boot(pred, gap, rng)]}
        print(f"{name}: {out[name]}")

    # the speaker's own labels vs those she checked from AI suggestions, as quoted in the paper
    out["by_protocol"] = {}
    for protocol, g in done.groupby("protocol"):
        keep = g[g.label >= 1]
        ids = set(keep.xh_id) | set(keep.en_id)
        out["by_protocol"][protocol] = dict(shares(g), gap_label_1_or_2=round(float(ba(pred) - ba(pred[pred.chunk_id.isin(ids)])), 3))
    print("by protocol:", out["by_protocol"])

    OUT.write_text(json.dumps(out, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
