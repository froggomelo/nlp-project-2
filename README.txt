# NLP Project 2: A Translation Floor for Cross-Lingual Archives: Spelling and Content Analysis in Imvo Zabantsundu

This code asks whether the isiXhosa and English sections of Imvo Zabantsundu (1884–1894) differ in content once the language signal is removed from a multilingual encoder's embeddings, and whether modernising the 1880s isiXhosa spelling changes the answer. `src/` holds the code, `colab/` the notebook that produced the reported embeddings and results, `data/` the data card and issue splits, `results/` the saved outputs the report uses, and `report/` the report (`report/report.pdf`).

## Setup

Run all commands from the project folder, with Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Embedding needs a GPU in practice, and everything else runs on a laptop CPU. The package versions used are recorded in `results/environment_analyse.json`.

## Data

Unzip `TxtLangFiles1870-1894_TB.zip` (from Moodle) into `data/raw/` so that these folders exist:

```
data/raw/TxtLangFiles1870-1894_TB/XhTxt
data/raw/TxtLangFiles1870-1894_TB/EnTxt
```

The modern isiXhosa corpora (Vukuzenzele, isiXhosa Wikipedia and OPUS-100 en-xh) and the encoders download automatically from Hugging Face.

## Quick test

After stage 1 below, this checks that every stage runs, without a GPU:

```bash
python src/smoke_test.py
python src/smoke_test.py --clean
```

It uses stand-in embeddings, so its numbers mean nothing. The second command removes its outputs.

## Running the experiments

| Stage | Command | What it gives | Time |
|---|---|---|---|
| 1. Text preparation | `python src/run_all.py prepare` | Cleans and chunks the text (60 words), flags repeated text, splits issues 70/15/15 by year, builds the modern word list and spelling variants, measures tool fit, and runs the simulation. | About 5 minutes on a CPU |
| 2. Embeddings | `python src/make_colab_bundle.py`, then `colab/embed_on_kaggle.ipynb` | Uploads `colab/colab_bundle.zip` to the notebook, which embeds every variant. Unzip the downloaded `embeddings_all.zip` into `data/embeddings/`. The notebook also runs stage 3. | About 1 hour on a free Kaggle Tesla T4 GPU |
| 3. Analysis | `python src/run_all.py analyse --threshold 0.55` and `python src/run_all.py analyse --threshold 0.55 --model-tag multilingual-e5-base` | Translated pairs, language removal, the section classifier with `k` and `C` chosen on validation issues, spelling comparisons with paired bootstrap CIs and the H2 verdict, and topic clusters. | About 15 minutes per encoder on a CPU |
| 4. Post-hoc checks | The five commands below | Floors on other pair sets, the topic-only rule, where errors fall, and pair outcomes. These need no retraining. | Minutes on a CPU |

```bash
python src/posthoc_checks.py . > results/posthoc_checks.txt
python src/floor_checks.py > results/floor_checks.txt
python src/topic_errors.py > results/topic_errors.txt
python src/pair_cases.py . > results/pair_cases.md
python src/report_figures.py
```

The last command draws `report/figures/fig_results.pdf` and `fig_topics.pdf`. LaBSE uses a pair threshold of 0.55. E5 uses 0.857, which keeps the same share of its candidates, because its similarities sit in a narrower range. On a machine with a GPU, stage 2 can also run directly:

```bash
python src/embed.py --variant raw
python src/embed.py --variant modern --chunks data/processed/chunks_modern.parquet --reuse raw
```

Do the same for `modern_asp` and `modern_prefix`. For the second encoder, add `--model intfloat/multilingual-e5-base --prefix "query: "`, for `raw` and `modern` only.

## Outputs and figures

| Report result | Saved in |
|---|---|
| Figure 1a, simulation | `results/synthetic/synthetic.json` |
| Figure 1b, Table S1, H2 | `results/compare/compare.md`, and `results/compare__multilingual-e5-base/compare.md` for E5 |
| Main result, floor, `k` and `C` | `results/raw/stage1.json` |
| Pair outcomes (201, 83, 27) | `results/pair_cases.md` |
| Hand-checked pairs | `results/raw/pair_verification.json` |
| Stricter pairs | `results/floor_checks.txt` |
| Pairs matched on names and numbers | `results/posthoc_checks.txt`, check 5 |
| E5 encoder and kNN | `results/raw__multilingual-e5-base/stage1.json` |
| Topic-only rule, 59%, 88%, 0.73 | `results/topic_errors.txt` |
| Figure S1, topics | `results/raw/qualitative.json` |
| Words rewritten, 191 names | `results/normaliser/modern_summary.json` |
| Missing words, tokens per word | `results/tool_metrics/tool_metrics.csv` |
| Modern word list sources | `results/lexicon/lexicon_sources.json` |

The pipeline also writes extra plots and large intermediate files, which are not stored here.

## Hand labels

Of 100 sampled test pairs, a fluent isiXhosa speaker labelled 24 herself. She also checked all 27 that a team member had labelled with AI translations and suggested labels, changing the ones she disagreed with. Labels were 2 (translation), 1 (partial) and 0 (different). The labelling sheets are not included. Their scores are saved in `results/raw/pair_verification.json`, and `src/verify_pairs.py` is the code that produced them, but it needs the sheets to run. The rest of the pipeline needs no labels.

## Key design decisions

- Splits are by issue, so no issue leaks between train, validation and test.
- Chunks in the wrong language, and repeated text (at least 30% of 5-word sequences seen in at least 5 issues), are left out through the `usable` column.
- Cleaning never changes spelling. Spelling changes only in `src/normalise.py`, on purpose.
- Language removal is learned from training pairs only, because plain mean-centring also removes average content differences.
- The normaliser rewrites a word only if the old spelling is almost absent from the modern word list (at most 2 uses) and the new one is common in it (at least 5). These thresholds were fixed in advance to favour precision, because over-correcting authors' spelling is the costly error.
- Topic, stricter-pair and pair-outcome analyses were performed after the main results, so they are exploratory, and no setting was chosen from test results.
