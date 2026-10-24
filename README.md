# newsindex

A Python library for building news-based Economic Policy Uncertainty (EPU) indices from newspaper archives, implementing the method behind the CAPE Africa EPU Index (Baker, Bloom and Davis, 2016, adapted for multilingual African press).

`newsindex` turns a collection of newspaper articles into monthly country, regional, and Africa-wide EPU series: classify articles, deduplicate them, fit a frozen baseline, score new months against it, and aggregate country series into GDP-PPP-weighted regional indices. Every number can be traced back to a documented function and its inputs.

## Scope (v1.0)

| In scope | Out of scope (later or separate) |
| --- | --- |
| EPU only: Economic AND Policy AND Uncertainty | ETU, TPU, and Energy Policy Uncertainty (planned) |
| Multilingual keyword matching and dictionary validation | Web scraping and newspaper collection |
| Article deduplication | Monthly scheduling, retries, cloud-storage promotion |
| Country index with a frozen baseline | Dashboards and charts |
| Regional and Africa aggregation with GDP-PPP weights | Distribution of article text or newspaper registries |

## Install

```bash
pip install -e .
```

Requires Python 3.10+. The only runtime dependencies are `pandas` and `numpy`.

## Usage

```python
import newsindex as ni

kw = ni.dictionary.load("data/eptu_country_keywords.json")
arts = ni.read_articles("Scrapped_Top_Articles/")          # site, date, title, url, text
counts = ni.index.count(ni.dedup.dedupe(arts), kw)          # country-site-month counts

baseline = ni.index.fit_baseline(counts, start="2015-01", end="2025-12")
baseline.save("baseline_2015_2025.json")

country = ni.index.score(counts, baseline)                  # country EPU series
gdp = ni.aggregate.monthly_gdp("data/gdp_ppp_2015_2025.csv", scale=1000)  # -> USD millions
africa = ni.aggregate.aggregate(country, gdp, groups=ni.AFRICA_REGIONS, baseline=baseline)
```

## Modules

| Module | Responsibility | Main functions |
| --- | --- | --- |
| `match` | Normalize text (Unicode NFKC, case-fold, quotes, dashes) and match whole words/phrases in any script | `normalize()`, `find_terms()`, `is_epu()` |
| `dictionary` | Load and validate country keyword files; enforce disjoint components; produce an audit and a fingerprint | `load()`, `validate()`, `fingerprint()` |
| `dedup` | Remove duplicate articles within a country-month by canonical URL, or by title/text when the URL is missing | `canonical_url()`, `dedupe()` |
| `index` | Count EPU articles per newspaper-month, fit the frozen baseline, score new months | `count()`, `fit_baseline()`, `score()` |
| `aggregate` | Country z-scores, GDP weights, regional and Africa series; annual-to-monthly GDP | `monthly_gdp()`, `aggregate()` |

`newsindex.read_articles(root)` reads the one input these functions don't take as a table: a directory of scraped article archives.

## Method

- **Classify.** An article counts as EPU if its title or body contains at least one Economic, one Policy, and one Uncertainty term from its country's dictionary. A term belongs to one component only.
- **Scale.** `S_it = EPU articles_it / all articles_it` for newspaper *i* in month *t*.
- **Standardise.** `N_it = S_it / sigma_i`, where `sigma_i` is the newspaper's population standard deviation over the baseline window (default 2015-01 to 2025-12).
- **Average and normalise.** `M_t` is the mean of `N_it` across newspapers; `EPU_t = 100 * M_t / mu`, where `mu` is the baseline mean of `M_t`.
- **Freeze.** `sigma_i`, `mu`, and the retained newspapers are saved as a JSON baseline (`Baseline.save`/`Baseline.load`); later months are scored against it and never recalibrated.
- **Aggregate.** `z_ct = (EPU_ct - m_c) / s_c`; regional `Z_rt = sum(w_ct * z_ct)` with GDP-PPP weights; rescaled to `100 + 10 * standardised Z_rt`. The Africa series applies the same steps across all countries directly — it is not an average of the regional series.

## Examples

See [`examples/`](examples/) for a hand-labeled test dataset (`sample_articles.xlsx`) and two notebooks that run the full pipeline against it (`test_with_excel.ipynb`) and against the real Drive archive (`test_with_drive.ipynb`).

## Data

- `data/eptu_country_keywords.json` — the production EPTU-2.0 keyword dictionary covering all 54 African countries (Economic, Policy, Trade, and Uncertainty term lists; `newsindex` uses the Economic/Policy/Uncertainty components for EPU). A term is matched only once it has been resolved against the other components, so the same literal phrase never satisfies two of them.
- `data/gdp_ppp_2015_2025.csv` — annual GDP (USD billions) by country, 2015–2025, used by `aggregate.monthly_gdp()`.

Article archives are not bundled; `read_articles()` expects the scraper's own layout: `<root>/<country>/<newspaper>/<year-month>/articles_<newspaper>_<year-month>.csv(.gz)`, each file with `site, date, title, url, text` columns.

## Design principles

- **Pure and path-free.** Functions take tables and return tables; no hard-coded folders or prompts. `read_articles()` is the one documented exception.
- **Reproducible.** The same inputs and baseline always give the same index; `dictionary.fingerprint()` lets a baseline record exactly which dictionary produced it.
- **One rule everywhere.** Deduplication and classification are applied identically to baseline and monthly updates.
- **Configurable groups.** Regions are passed in as a mapping; `AFRICA_REGIONS` is provided as the default (CAPE's five-region grouping).
- **Light dependencies.** Python 3.10+, pandas, and numpy only.

## Testing

```bash
pip install -e ".[test]"
pytest
```

Tests cover Unicode/multilingual term matching (including Arabic script and hyphen/space phrase equivalence), dictionary validation against the real production dictionary, dedup behaviour, baseline fitting and scoring against hand-computed values, GDP log-linear interpolation, and a full pipeline smoke test.

## Roadmap

| Version | Content |
| --- | --- |
| 1.0 | EPU: deduplication, country index, frozen baseline, aggregation, documentation and tests |
| 1.1 | Trade indices (ETU, TPU) through a general index rule |
| 1.2 | Energy Policy Uncertainty (EgPU); optional command-line tool |

## License

MIT. See [LICENSE](LICENSE).
