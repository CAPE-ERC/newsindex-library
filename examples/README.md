# Examples

| File | What it is |
| --- | --- |
| [`sample_articles.xlsx`](sample_articles.xlsx) | 24 hand-labeled articles (2 countries x 2 newspapers x 2 months x 3 articles) for testing the pipeline without needing the real archive. Every `expected_is_epu` label is verified against the real production dictionary; the `note_for_reviewer` column names exactly which Economic/Policy/Uncertainty terms each title hits. |
| [`test_with_excel.ipynb`](test_with_excel.ipynb) | Runs `sample_articles.xlsx` through classification, counting, baseline fitting, scoring, baseline save/load, and aggregation -- and asserts every result against hand-computed values. Self-contained; needs no external data. Already executed in-place, so you can read the outputs without re-running it. |
| [`test_with_drive.ipynb`](test_with_drive.ipynb) | Runs the same pipeline against the real scraped-article archive in Google Drive (`Shak_Scrapped_Top_Articles`). Works in Colab (mounts Drive) or locally against a Google Drive for Desktop sync path. Defaults to a small country slice -- widen `COUNTRIES` deliberately once you know the runtime on your connection. |

Both notebooks assume they're run from inside a clone of this repo, so they can reach `data/eptu_country_keywords.json` and `data/gdp_ppp_2015_2025.csv` by relative path (those files aren't bundled in the PyPI package).
