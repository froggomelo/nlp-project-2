"""Hand check of the spelling rewrites (no retraining, CPU, seconds).

100 rewrites that the modernising rules made were sampled by rule and by how often each occurs
(data/hand_checks/rewrites_key.csv). An isiXhosa speaker marked each one correct (1) or not (0) in its
context, leaving blank the ones she was unsure of (data/hand_checks/rewrites_checked_Endinako.xlsx).

Reports the share correct among the rewrites she judged, by rule, and weighted back to the running text:
within each frequency band, by how often each sampled rewrite occurs, then across bands by their token counts.

Usage:
    python -m src.posthoc.rewrite_check      # -> results/raw/rewrite_check.json
"""
import json
from pathlib import Path

import pandas as pd

SHEET = Path("data/hand_checks/rewrites_checked_Endinako.xlsx")
KEY = Path("data/hand_checks/rewrites_key.csv")
OUT = Path("results/raw/rewrite_check.json")


def main():
    marks = pd.read_excel(SHEET, sheet_name="Rewrites", dtype={"correct": "string"})
    d = pd.read_csv(KEY, encoding="utf-8-sig").merge(marks[["row_id", "correct", "gold", "note"]], on="row_id")
    d["correct"] = pd.to_numeric(d["correct"], errors="coerce")
    judged = d[d.correct.notna()]

    def token_weighted(g):
        # each band: occurrence-weighted share correct among judged rewrites, weighted by the band's tokens
        g = g.assign(right=g["count"] * g.correct)
        bands = g.groupby("bin").agg(right=("right", "sum"), seen=("count", "sum"), tokens=("tokens_in_bin", "first"))
        share = bands.right / bands.seen
        return float((share * bands.tokens).sum() / bands.tokens.sum())

    out = {"sampled": int(len(d)), "judged": int(len(judged)), "left_blank": int(d.correct.isna().sum()),
           "correct": int(judged.correct.sum()), "share_correct": round(float(judged.correct.mean()), 3),
           "token_weighted_share_correct": round(token_weighted(judged), 3),
           "by_rule": {r: {"judged": int(len(g)), "correct": int(g.correct.sum())} for r, g in judged.groupby("rule")},
           "marked_wrong": judged[judged.correct == 0][["row_id", "old", "new", "gold"]].to_dict("records"),
           "left_blank_rows": d[d.correct.isna()][["row_id", "old", "new"]].to_dict("records")}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
