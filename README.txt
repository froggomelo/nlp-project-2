===================================================================================================
NLP Project 2: A Translation Floor for Cross-Lingual Archives:
Spelling and Content Analysis in Imvo Zabantsundu
COMS4054A
===================================================================================================

This code asks whether the isiXhosa and English sections of Imvo Zabantsundu (1884-1894) differ in
content once the language signal is removed from LaBSE embeddings, and whether modernising the
1880s isiXhosa spelling changes that answer. The newspaper's own translations carry the same
content in both languages, so a section classifier's accuracy on them (the floor) shows how well it
separates the sections by language alone; only accuracy above the floor counts as a content
difference.

  src/              the pipeline
  colab/            the GPU notebook (embed_on_colab.ipynb; the others are records of earlier runs)
  labels/           the hand-labelling sheets and guide
  report/           the report
  docs/             supporting write-ups and team notes
  data/processed/   the data card and splits
  results/          everything the pipeline writes


---------------------------------------------------------------------------------------------------
SETUP
---------------------------------------------------------------------------------------------------

Run every command from the project folder, with Python 3.11 or newer.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Embedding needs a GPU in practice (about an hour on a free Colab T4); every other stage runs on a
laptop CPU. Activate the environment again with `source .venv/bin/activate` in every new terminal.


---------------------------------------------------------------------------------------------------
DATA
---------------------------------------------------------------------------------------------------

Unzip the course file TxtLangFiles1870-1894_TB.zip into data/raw/ so that these folders exist:

    data/raw/TxtLangFiles1870-1894_TB/XhTxt
    data/raw/TxtLangFiles1870-1894_TB/EnTxt

The modern isiXhosa corpora (Vukuzenzele, isiXhosa Wikipedia, OPUS-100 en-xh) download from
Hugging Face in step 1. The raw text, the embeddings and every generated .parquet file are not
committed, so a fresh clone must rebuild them with the steps below.


---------------------------------------------------------------------------------------------------
QUICK TEST (no GPU)
---------------------------------------------------------------------------------------------------

    python src/smoke_test.py
    python src/smoke_test.py --clean

The first command runs every stage after text preparation on stand-in embeddings, to check that the
code runs and the files line up; its numbers mean nothing. It needs the outputs of step 1. Its
outputs go to results/*__FAKE/ and data/embeddings/*_FAKE.npy, and --clean removes them.


---------------------------------------------------------------------------------------------------
PIPELINE
---------------------------------------------------------------------------------------------------

  Step                   Where      Time
  ---------------------  ---------  --------------------
  1. Text preparation    CPU        ~5 min
  2. Hand-labelling      people     ~3-4 h (optional)
  3. Embeddings          GPU        ~1 h on a Colab T4
  4. Analysis            CPU        ~15 min per encoder

1. Text preparation

    python src/run_all.py prepare

   This runs, in order:
     prepare_data.py      clean, chunk (60 words), flag boilerplate, split by issue (70/15/15)
     build_lexicon.py     modern isiXhosa word list with per-source counts; records each dataset's
                          Hugging Face revision and token counts
     normalise.py         candidate rewrites; tuning (if labels exist); the variants all, asp,
                          prefix and keepnames (both rules, but likely names keep their 1880s
                          spelling)
     tool_metrics.py      tokens per word (LaBSE, XLM-R) and out-of-vocabulary rate per variant
     synthetic_check.py   validates the language-removal method on data with a known answer, with
                          clean pairs and with 40% false pairs (as in the real candidates)

2. Hand-labelling (see labels/LABELLING_GUIDE.md)

    python src/normalise.py sheet    ->  labels/normaliser_to_label.csv
                                         save as labels/normaliser_labelled.csv
    python src/pairs.py sheet        ->  labels/pairs_to_label.csv  (needs raw embeddings)
                                         save as labels/pairs_labelled.csv

   After labelling the normaliser sheet, rerun step 1 so the tuned thresholds are used. Neither
   sheet was labelled in full for the report, which uses the provisional pair threshold in step 4.
   Rerunning either sheet command overwrites the blank sheet.

3. Embeddings

    python src/make_colab_bundle.py  ->  colab/colab_bundle.zip

   Open colab/embed_on_colab.ipynb in Colab in the browser (downloads fail from the VS Code
   extension), choose a T4 GPU, run all cells, upload the bundle when asked, and unzip the
   downloaded embeddings_all.zip into data/embeddings/.

   On a machine with a GPU instead:

    python src/embed.py --variant raw
    python src/embed.py --variant modern --chunks data/processed/chunks_modern.parquet --reuse raw

   and the same for modern_asp and modern_prefix. For the second encoder add
   --model intfloat/multilingual-e5-base --prefix "query: ".

4. Analysis (LaBSE first, then the second encoder)

    python src/run_all.py analyse --threshold 0.55
    python src/run_all.py analyse --threshold 0.55 --model-tag multilingual-e5-base

   This runs, in order:
     pairs.py             mutual-nearest-neighbour pairs per issue
     classify.py          section classifier with the language subspace (from translated pairs)
                          removed; k chosen so the validation floor is at chance, C tuned on
                          validation; also reports the gap for a kNN probe
     compare.py           raw vs modern vs ablations: all / floor / gap, alignment, paired
                          bootstrap CIs, and an H2 verdict against a +/-0.02 margin
     qualitative.py       topic clusters, isiXhosa-only topics, per-issue divergence against a
                          permutation baseline

   --threshold 0.55 is the pair-similarity threshold the report uses for LaBSE. E5's similarities
   sit in a narrower range, so it gets the threshold that keeps the same share of its candidates
   (0.857). If labels/pairs_labelled.csv exists, the threshold is tuned on it instead. The LaBSE
   run also repeats the raw classifier with stricter pairs (--robust-threshold, default 0.65) into
   results/raw_thr0.65/.

   Expected notice, not an error: "multilingual-e5-base: no embeddings for [...]". Only raw and
   modern are embedded with E5; the ablations and keepnames run on LaBSE only.

   Post-hoc checks and report figures, from the saved results:

    python src/posthoc_checks.py . > results/posthoc_checks.txt     (~2 min, no retraining)
    python src/report_figures.py     ->  report/figures/fig_results.pdf, fig_topics.pdf

   The checks are the floor by pair similarity, the gap without number-heavy chunks, and
   separability by period. Both scripts read .parquet files that step 4 writes and git ignores.


---------------------------------------------------------------------------------------------------
OUTPUTS
---------------------------------------------------------------------------------------------------

  data/processed/data_card.md            corpus statistics per year and split
  results/normaliser/                    thresholds, what each variant changed, name rewrites
  results/tool_metrics/                  tokens per word and OOV rate per variant and year
  results/lexicon/lexicon_sources.json   modern corpora: revisions, token counts, word-list share
  results/synthetic/                     method validation: fig_synthetic.png,
                                         fig_synthetic_noisy.png (40% false pairs)
  results/raw/                           stage1.json, fig_k_curve.png, qualitative.md,
                                         fig_clusters.png
  results/raw_thr0.65/                   the raw classifier with stricter pairs
  results/modern*/                       the classifier on each spelling variant
  results/compare/                       compare.md, fig_compare.png (central figure)
  results/*__multilingual-e5-base/       the same for the second encoder
  results/posthoc_checks.txt             the post-hoc checks
  results/environment_<phase>.json       Python and package versions of each run


---------------------------------------------------------------------------------------------------
KEY DESIGN DECISIONS
---------------------------------------------------------------------------------------------------

  - Splits are by issue, so no issue leaks between train, validation and test.
  - Boilerplate (>= 30% of 5-word sequences seen in >= 5 issues) and language-mismatched chunks are
    excluded through the `usable` column.
  - Cleaning never changes spelling; spelling changes only in normalise.py, on purpose.
  - Language removal uses only translated pairs, whose content is identical. Plain mean-centring is
    reported for comparison, because it also removes genuine content differences
    (synthetic_check.py).
  - Only accuracy above the translated-pair floor counts as a content difference.
  - The normaliser rewrites a word only if the original is unattested in modern text and the
    rewrite is attested, and joins a prefix only if the joined form is attested. The thresholds
    (2, 5) are fixed in advance to favour precision, because over-correcting the authors' spelling
    is the costly error; `normalise.py tune` replaces them (by validation F0.5) if the sheet is
    labelled.


---------------------------------------------------------------------------------------------------
FURTHER DOCUMENTATION
---------------------------------------------------------------------------------------------------

  labels/LABELLING_GUIDE.md      the two labelling jobs and how to judge each row
  docs/HANDOVER.txt              team notes: rebuilding from a clone, git workflow, troubleshooting
  docs/findings.pdf              longer account of the results; docs/make_findings_figures.py
                                 draws its figures
  docs/results_explained.pdf     the results explained step by step
  report/report.pdf              the report
