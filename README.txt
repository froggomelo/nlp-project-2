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
       normalise.py      candidate rewrites; tune (if labels exist); apply all / asp / prefix variants and
                         keepnames (both rules, but likely proper names keep their 1880s spelling)
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
     Post-hoc checks on the test predictions (CPU, ~2 min; no retraining):
       python src/posthoc_checks.py . > results/posthoc_checks.txt
       = floor by pair similarity, gap without number-heavy chunks, separability by year
     Report figures from saved results (CPU, ~10 s):
       python src/report_figures.py   -> report/figures/fig_results.pdf, fig_topics.pdf
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
    attested, and joins a prefix only if the joined form is attested. The thresholds (2, 5) are fixed in
    advance to favour precision, because over-correcting authors' spelling is the costly error; all three
    rule families are on the hand-checking sheet, and `normalise.py tune` replaces the thresholds (by
    validation F0.5) if a fluent reader labels it.
  - Alignment across spelling variants is measured on embedding-free (anchor) pairs once pairs are labelled;
    until then compare.py uses pairs selected in raw space, which favours raw, and says so.

CHANGES ON 21 OCTOBER 2026 (team note: delete this section before submission)
  What changed
    1. classify.py      C grid widened to 0.01-1000; warns if the chosen C sits at the edge of the grid.
    2. pairs.py match   the second encoder (E5) gets its own provisional pair threshold: the one that
                        keeps the same share of its candidates as LaBSE keeps at 0.55. E5 similarities
                        mostly lie between 0.83 and 0.88, so a threshold of 0.55 filtered nothing.
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

  Colab run after these changes (later on 21 October)
    Steps 1-3 of the old next-steps list are done: the bundle was rebuilt with the new normaliser and
    every variant re-embedded. The new normaliser rewrites 9.1% of isiXhosa tokens (was 12.4%):
    aspiration 289,816 tokens (7.8%, unchanged), prefix joins before names 47,517 (was 169,556),
    prefix joins before lowercase words 797. Raw LaBSE pair candidates are identical to the earlier
    run, and E5's matched threshold is 0.857 (keeps 24.6% of its candidates).
    The analysis then failed: classify.py was killed by Colab for running out of memory
    (SIGKILL 9), for both LaBSE and E5, so no stage1.json, compare or qualitative results exist yet.

  Out-of-memory bug fixed
    9. classify.py      kept a full copy of the embeddings for each of the 9 methods, and every
                        logistic-regression fit left a ~0.3 GB float64 copy of the training data in
                        reference cycles; across 54 fits this reached ~13 GB. Each method is now scored
                        on validation and test straight away and its copy freed, and garbage is
                        collected after every fit. Peak memory at full size (68,392 x 768): 13.1 GB ->
                        2.2 GB, with identical results; about one minute per classifier run.
    10. run_all.py      the 0.65 robustness run can no longer stop the pipeline: if it fails, a
                        warning is printed and compare.py and qualitative.py still run.
    11. synthetic_check.py  figures redrawn at the report's column width (3.4 in) so labels print
                        legibly; same numbers.
    Tested end to end with the package versions Colab used (scikit-learn 1.6.1, pandas 2.2.3,
    numpy 2.1.3), including a full-size stand-in dataset for the memory check.

  Expected notice, not an error
    "multilingual-e5-base: no embeddings for ['modern_asp', 'modern_keepnames', 'modern_prefix'];
    analysing ['raw', 'modern']". The notebook embeds only raw and modern with E5 (the robustness
    check); the ablations and the keepnames variant run on LaBSE only.

  Fixes after a code check (later on 21 October)
    12. normalise.py    rewrites now keep the author's capitalisation inside a word (uTixo -> uThixo,
                        was uthixo; 4 tokens in the corpus); the labelling sheet now finds an example
                        context for every word (it searched only 20,000 texts before).
    13. qualitative.py  notes when the best number of topic clusters is the largest tried (32 was).
    14. smoke_test.py   passed an explicit pair threshold: the stand-in embeddings' similarities are
                        below the tuning grid, so the test stopped at classify.py. It now runs end to end.
    The fix in 12 changes 4 tokens, so the next Colab run should move the reported numbers negligibly.

  New analyses (later on 21 October)
    15. normalise.py --rules keepnames: a fifth variant that modernises everything except likely names
        (capitalised in >= 90% of uses), to test whether the alignment gain survives without rewriting
        names of peoples, places and God. Added to run_all.py, compare.py, tool_metrics.py, the smoke
        test and the Colab notebook (embedded with LaBSE only). Not run: the paper mentions it only as a
        test the code supports.
    16. posthoc_checks.py check 4: the gap per period (1884-89, 1890-94), each against its own floor.
        Separability rises over the decade, but so does the floor; the gap does not grow (LaBSE change
        -0.012, 95% CI -0.073 to +0.048). The paper now says the rise is language, not content.

  Final report edits (evening of 21 October)
    17. normalise.py    capitalised_share leaves out ALL-CAPS uses: they say nothing about whether a word
                        is a name, and counting them hid Tixo (85% -> 99% capitalised). Likely names
                        rewritten by aspiration: 191 words, 19,917 rewrites (was 149 / 12,966).
    18. Text side rerun locally from the course corpus (prepare_data, build_lexicon, normalise,
        tool_metrics; no embedding or classifier runs). Chunks, splits and the data card are identical to
        the Colab bundle's, and the normaliser counts match the Colab run exactly (289,816 / 797 / 47,517).
        New: results/lexicon/lexicon_sources.json (3,329,042 tokens; 88.5% OPUS-100) and
        results/tool_metrics for the current normaliser (modern: 24.8% OOV, 2.10 LaBSE tokens per word).
    19. Label-free pair check (from results/raw/pair_candidates.parquet): kept pairs share >= 3 names or
        numbers in 25.7% of cases against 2.9% of dropped ones.

  Next steps, in order
    The report is complete without further runs. Optional, if time allows:
    1. Colab Run all with a fresh bundle (python src/make_colab_bundle.py) adds the keepnames row.
    2. Labelling translated pairs (labels/LABELLING_GUIDE.md) would replace the provisional 0.55
       threshold; the paper would then need its "not hand-labelled" sentences updated.
