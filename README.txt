Reading Both Halves of Imvo: the consequences of modernising 1880s isiXhosa
COMS4054A/7062A/7066A NLP Project 2

QUESTION
  Do the isiXhosa and English sections of Imvo Zabantsundu (1884-1894) differ in content once the
  language signal is removed, and how does modernising the 1880s isiXhosa spelling change that answer?

SETUP
  python -m pip install -r requirements.txt          (Python 3.11+)
  Embedding needs a GPU in practice (~1 h on a free Colab T4); everything else runs on a laptop CPU.

DATA
  Unzip TxtLangFiles1870-1894_TB.zip into data/raw/ so that these folders exist:
    data/raw/TxtLangFiles1870-1894_TB/XhTxt
    data/raw/TxtLangFiles1870-1894_TB/EnTxt
  Modern isiXhosa corpora (Vukuzenzele, isiXhosa Wikipedia, OPUS-100 en-xh) download automatically
  from Hugging Face in step 1.

TEAM NOTES
  Rebuilding from a git clone, git workflow, labelling order and troubleshooting: docs/HANDOVER.txt

PIPELINE (run from the project root)
  1. Text preparation (CPU, ~5 min):
       python src/run_all.py prepare
     = prepare_data.py   clean, chunk (60 words), flag boilerplate, split by issue (70/15/15 by year)
       build_lexicon.py  modern isiXhosa word list with per-source counts; records each dataset's
                         Hugging Face revision and token counts in results/lexicon/lexicon_sources.json
       normalise.py      candidate rewrites; tune (if labels exist); apply all / asp / prefix variants
       tool_metrics.py   tokens per word (LaBSE, XLM-R) and out-of-vocabulary rate per variant
       synthetic_check.py  validates the language-removal method on data with a known answer, with clean
                           pairs and with 40% false pairs (as in the real candidates)

  2. Hand-labelling (people, ~3-4 h in total; see labels/LABELLING_GUIDE.md):
       python src/normalise.py sheet         -> labels/normaliser_to_label.csv  (save as normaliser_labelled.csv)
       python src/pairs.py sheet              -> labels/pairs_to_label.csv       (save as pairs_labelled.csv;
                                                 needs raw embeddings from step 3)
     After labelling the normaliser sheet, rerun step 1 so the tuned thresholds are used.

  3. Embeddings (GPU):
       python src/make_colab_bundle.py       -> colab/colab_bundle.zip
       open colab/embed_on_colab.ipynb in Colab (browser), upload the bundle, run all cells,
       unzip the downloaded embeddings_all.zip into data/embeddings/.
     On a machine with a GPU instead:
       python src/embed.py --variant raw
       python src/embed.py --variant modern --chunks data/processed/chunks_modern.parquet --reuse raw
       (same for modern_asp and modern_prefix; add --model intfloat/multilingual-e5-base
        --prefix "query: " for the second encoder)

  4. Analysis (CPU, ~15 min per encoder):
       python src/run_all.py analyse
       python src/run_all.py analyse --model-tag multilingual-e5-base
     = pairs.py        mutual-nearest-neighbour pairs per issue; threshold tuned on labelled pairs
       classify.py     section classifier with the language subspace (from translated pairs) removed;
                       k chosen so the validation pair floor is at chance, C tuned on validation;
                       modern variants reuse raw's reference pairs and k; also reports the gap for a
                       kNN probe (a linear probe cannot see differences that mean-centring leaves)
       compare.py      raw vs modern vs ablations: all / floor / gap, alignment, paired bootstrap CIs,
                       and an H2 verdict (equivalent / different / inconclusive) against a +/-0.02 margin
       qualitative.py  topic clusters, isiXhosa-only topics, per-issue divergence vs permutation baseline
     Without labels yet, add --threshold 0.55 for a provisional run. That threshold applies to LaBSE;
     the second encoder gets the threshold that keeps the same share of its candidates (pairs.py
     match), because its similarities sit in a different range. The LaBSE run also repeats the raw
     classifier at a stricter threshold (--robust-threshold, default 0.65) into results/raw_thr0.65.
     Each run records the Python and package versions in results/environment_<phase>.json.

  Check that everything runs without a GPU (stand-in embeddings; numbers meaningless):
       python src/smoke_test.py   then   python src/smoke_test.py --clean

OUTPUTS
  data/processed/data_card.md            corpus statistics
  results/normaliser/                    thresholds, what changed, likely-name rewrites
  results/tool_metrics/                  fertility and OOV per variant and year
  results/lexicon/lexicon_sources.json   modern corpora: revisions, token counts, share of the word list
  results/synthetic/                     method validation: fig_synthetic.png, fig_synthetic_noisy.png
  results/raw/                           Stage 1: stage1.json, fig_k_curve.png, qualitative.md, fig_clusters.png
  results/raw_thr0.65/                   Stage 1 robustness: the raw classifier with stricter pairs
  results/compare/                       Stage 3: compare.md, fig_compare.png (central figure)
  results/*__multilingual-e5-base/       the same for the second encoder

KEY DESIGN DECISIONS
  - Splits are by issue, so no issue leaks between train, validation and test.
  - Boilerplate (>= 30% of 5-word sequences seen in >= 5 issues) and language-mismatched chunks are
    flagged and excluded through the `usable` column.
  - Cleaning never changes spelling; spelling changes only in normalise.py, on purpose.
  - Language removal uses only translated pairs, whose content is identical; plain mean-centring is
    reported for comparison because it also removes genuine content differences (synthetic_check.py).
  - Only accuracy above the translated-pair floor counts as a content difference.
  - The normaliser rewrites a word only if the original is unattested in modern text and the rewrite is
    attested, and joins a prefix only if the joined form is attested; thresholds favour precision (F0.5)
    because over-correcting authors' spelling is the costly error. All three rule families are hand-checked.
  - Alignment across spelling variants is measured on embedding-free (anchor) pairs to avoid selection bias.

CHANGES ON 21 OCTOBER 2026 (team note: delete this section before submission)
  What changed
    1. classify.py      C grid widened to 0.01-1000; warns if the chosen C sits at the edge of the grid.
    2. pairs.py match   the second encoder (E5) gets its own provisional pair threshold: the one that
                        keeps the same share of its candidates as LaBSE keeps at 0.55. E5 similarities
                        all lie between 0.83 and 0.87, so a threshold of 0.55 filtered nothing.
    3. run_all.py       the LaBSE run repeats the raw classifier with stricter pairs (0.65) into
                        results/raw_thr0.65/, to check the gap does not depend on weak pairs.
    4. normalise.py     prefix joins before capitalised words must pass the word-list check (the joined
                        form must be attested), and they are now on the labelling sheet and in tuning.
                        Before, they were applied to every match without a check (4.6% of tokens).
    5. classify.py      also reports the gap for a kNN probe (no removal, mean-centring, chosen k):
                        a linear probe cannot see differences that mean-centring leaves.
    6. synthetic_check.py  second condition with 40% false pairs, as in the real candidates
                        (results/synthetic/fig_synthetic_noisy.png).
    7. compare.py       H2 verdict: equivalent / different / inconclusive, from the 90% CI of the gap
                        change against a +/-0.02 margin; "invalid" if a classifier predicts one class.
    8. Provenance       package versions -> results/environment_<phase>.json; Hugging Face revisions
                        and token counts of the modern corpora -> results/lexicon/lexicon_sources.json.
                        Team notes moved from this README to docs/HANDOVER.txt.

  Bug fixed (it predates these changes)
    In the full "modern" variant, aspiration ran before prefix joining, so a prefix before a word that
    aspiration had rewritten was never joined ("ku Rulumente" became "ku Rhulumente", not
    "kuRhulumente"). Join decisions are now made on the raw text, before aspiration.

  Testing
    Every stage was run end to end on a small fake corpus (prepare, normaliser, synthetic check, pairs,
    classifier, robustness run, matched E5 threshold, compare, qualitative). This shows the code runs;
    it says nothing about the real results.
    The synthetic check does not use the data, so its output in results/synthetic/ is final. Clean
    pairs: unchanged (pairs 0.097 vs oracle 0.099 at delta 0.4). With 40% false pairs the pair method
    still finds a positive gap but underestimates it (0.045 vs 0.059), and mean-centring still reports
    0: noisy pairs make the H1 estimate cautious, not inflated.

  Next steps, in order (whoever has the data)
    1. python src/run_all.py prepare        rebuilds the normaliser candidates under the new check
    2. python src/normalise.py sheet        regenerates the labelling sheet, now with prefix joins before
                                            names (the current sheet is blank, so nothing is lost)
    3. python src/make_colab_bundle.py, then Run all on Colab: modern and modern_prefix must be
       re-embedded because their text changed
    4. Label pairs and normaliser words, then one final python src/run_all.py analyse
    Expect the modern-variant numbers to move (the prefix rule now rewrites far fewer tokens); the raw
    H1 result is unaffected by the normaliser change.
