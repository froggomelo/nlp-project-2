"""Run the pipeline stages in order. Embedding (GPU-heavy) is a separate step; see README.txt.

Usage:
    python -m src.run_all prepare        # Stage 0 + Stage 2 text side: data, word list, normaliser variants
    python -m src.run_all analyse        # after embeddings exist: pairs, classifier, comparison, analyses
    python -m src.run_all analyse --model-tag multilingual-e5-base     # second encoder
Options:
    --threshold X   provisional LaBSE pair-similarity threshold if labels/pairs_labelled.csv is not
                    ready yet; other encoders get the threshold that keeps the same share of their
                    candidates (pairs.py match), because similarity scales differ between encoders
    --robust-threshold X   LaBSE only: rerun the raw classifier with this stricter pair threshold
                    (default 0.65) into results/raw_thr<X>, to check the gap does not hinge on weak pairs
Every run records the Python and package versions in results/environment_<phase>.json.
"""
import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

VARIANTS = ["raw", "modern", "modern_asp", "modern_prefix", "modern_keepnames"]
PACKAGES = ["pandas", "pyarrow", "numpy", "scipy", "scikit-learn", "torch", "transformers",
            "sentence-transformers", "huggingface_hub", "matplotlib"]


def sh(*args):
    print("\n$ python -m", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def record_environment(phase):
    def version(p):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return None
    env = dict(python=sys.version.split()[0], platform=platform.platform(), packages={p: version(p) for p in PACKAGES})
    Path("results").mkdir(exist_ok=True)
    Path(f"results/environment_{phase}.json").write_text(json.dumps(env, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "analyse"])
    ap.add_argument("--model-tag", default="LaBSE")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--robust-threshold", type=float, default=0.65)
    a = ap.parse_args()
    tag = ["--model-tag", a.model_tag]
    record_environment(a.phase)

    if a.phase == "prepare":
        sh("src.prepare.prepare_data")
        sh("src.prepare.build_lexicon")
        sh("src.prepare.normalise", "candidates")
        if Path("labels/normaliser_labelled.csv").exists():
            sh("src.prepare.normalise", "tune")
        for rules in ["all", "asp", "prefix", "keepnames"]:
            sh("src.prepare.normalise", "apply", "--rules", rules)
        sh("src.prepare.tool_metrics")
        sh("src.prepare.synthetic_check")
        return

    # only the variants embedded with this encoder (the Colab notebook embeds raw + modern for E5)
    variants = [v for v in VARIANTS if Path(f"data/embeddings/{v}_{a.model_tag}.npy").exists()]
    if "raw" not in variants:
        sys.exit(f"\nNo data/embeddings/raw_{a.model_tag}.npy: run the embedding step first (see README).")
    skipped = sorted(set(VARIANTS) - set(variants))
    if skipped:
        print(f"\n{a.model_tag}: no embeddings for {skipped}; analysing {variants}")
    for v in variants:
        sh("src.analysis.pairs", "candidates", "--variant", v, *tag)
    if Path("labels/pairs_labelled.csv").exists():
        sh("src.analysis.pairs", "tune", *tag)
        thr = []
    elif a.threshold is not None and a.model_tag == "LaBSE":
        thr = ["--threshold", str(a.threshold)]
        print(f"\nNo labelled pairs yet: using provisional threshold {a.threshold}")
    elif a.threshold is not None:
        if not Path("results/raw/pair_candidates.parquet").exists():
            sys.exit("\nRun the LaBSE analysis first: this encoder's provisional threshold is matched to LaBSE's.")
        sh("src.analysis.pairs", "match", *tag, "--ref-threshold", str(a.threshold))
        thr = []  # classify.py reads the matched threshold from pair_threshold.json
        print(f"\nNo labelled pairs yet: {a.model_tag} threshold matched to LaBSE at {a.threshold}")
    else:
        sh("src.analysis.pairs", "sheet", *tag)
        sys.exit("\nLabel labels/pairs_to_label.csv, save it as labels/pairs_labelled.csv and rerun, "
                 "or pass --threshold 0.55 as in the report.")
    sh("src.analysis.classify", "--variant", "raw", *tag, *thr)
    if a.model_tag == "LaBSE":
        used = json.loads(Path("results/raw/stage1.json").read_text())["threshold"]
        if a.robust_threshold > used:
            try:  # a robustness check must not stop the main analysis
                sh("src.analysis.classify", "--variant", "raw", *tag, "--threshold", str(a.robust_threshold),
                   "--run-name", f"raw_thr{a.robust_threshold}")
            except subprocess.CalledProcessError:
                print(f"\nWARNING: robustness run at threshold {a.robust_threshold} failed; continuing")
    for v in variants[1:]:
        sh("src.analysis.classify", "--variant", v, "--pairs-from", "raw", "--k-from", "raw", *tag, *thr)
    if len(variants) > 1:
        sh("src.analysis.compare", "--variants", *variants, *tag)
    sh("src.analysis.qualitative", *tag)


if __name__ == "__main__":
    main()
