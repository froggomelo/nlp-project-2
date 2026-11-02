Hand checks by isiXhosa speakers

Translated pairs (labels: 2 = translation, 1 = partial, 0 = different; blank = not sure)
  pairs_checked_Yonela.xlsx     51 sampled test pairs, final labels by Yonela Siko. She labelled 24 herself
                                and checked 27 that a team member had labelled with AI-suggested labels,
                                changing those she disagreed with.
  pairs_checked_Endinako.xlsx   20 further sampled test pairs labelled by Endinako Dyantyi; she left 4 blank because she was unsure of them.
  Each workbook has a Labels tab and the texts on isiXhosa, English and Side by side tabs; the .csv file
  of the same name holds the same rows (labels and both texts) in one table.
  All 67 labels, with who gave each and how (protocol column), are in data/pair_labels.csv, which the
  code reads: python -m src.posthoc.verify_pairs -> results/raw/pair_verification.json

Spelling rewrites (correct: 1 = right modern spelling in this context, 0 = not; gold = the right spelling)
  rewrites_checked_Endinako.xlsx  100 rewrites the modernising rules made, checked by Endinako Dyantyi; she left
                                  6 blank where she was unsure.
  rewrites_checked_Endinako.csv   the same rows as a CSV file.
  rewrites_key.csv                for each row: rule, original and rewrite, how often it occurs, and the
                                  sampling band, used to weight the result back to the running text.
  python -m src.posthoc.rewrite_check -> results/raw/rewrite_check.json
