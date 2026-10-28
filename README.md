# newsindex

A Python library for building news-based Economic Policy Uncertainty (EPU) indices from newspaper archives, implementing the method behind the CAPE Africa EPU Index (Baker, Bloom and Davis, 2016, adapted for multilingual African press).

`newsindex` turns a collection of newspaper articles into monthly country, regional, and Africa-wide EPU series: classify articles, deduplicate them, fit a frozen baseline, score new months against it, and aggregate country series into weighted regional indices (GDP-PPP by default, or a weight table or function of your own). Every number can be traced back to a documented function and its inputs.

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

## Quick start

`run_index`/`run_index_with_baseline` wrap reading, deduplicating, counting, and scoring into a single call, for anyone who just wants an index out of an article archive without assembling the pipeline by hand:

```python
import newsindex as ni

# No baseline: one window, fit and scored in-sample.
country = ni.run_index(
    "Scrapped_Top_Articles/", "data/eptu_country_keywords.json", start="2020-01", end="2020-12"
)

# With baseline: freeze 2015-2024, then score 2025 against it.
country, baseline = ni.run_index_with_baseline(
    "Scrapped_Top_Articles/",
    "data/eptu_country_keywords.json",
    windows=[("2025-01", "2025-12")],
    baseline_start="2015-01",
    baseline_end="2024-12",
)
baseline.save("baseline_2015_2024.json")  # reuse next run: pass baseline=... instead of baseline_start/end
```

Both print progress to stderr by default, one line per country-month as it's processed (`verbose=False` to silence), and accept `countries=[...]` to read only some of the archive's country folders (default: every folder; folder names are matched case-insensitively). Neither function has a bundled default dictionary -- `dictionary_path` is always an explicit argument, since this library ships no proprietary keyword data.

Both stream the archive internally via `index.count_archive()` -- see [Performance](#performance) -- rather than reading the whole requested scope into memory at once, so they're the memory-safe choice for a large archive. They compose `index.count_archive()` with `index.compute_index()`/`index.fit_baseline()`+`index.score_windows()`; reach for the lower-level functions directly (see Usage below) for anything a single call can't express, e.g. inspecting the deduplicated article table or reusing one count table across several baseline windows.

### Quick-call shortcuts

For the shapes of request that come up most often, a few named shortcuts sit on top of `run_index` -- same memory-safe streaming underneath, just naming the intent instead of assembling the arguments:

```python
ni.quick_index(root, dict_path, "Nigeria", 2020)                # one country, one calendar year
ni.index_all(root, dict_path, "2020-01", "2020-12")              # every country folder, explicitly
ni.index_countries(root, dict_path, ["Nigeria", "Kenya"], "2020-01", "2020-12")
ni.index_region(root, dict_path, "West", "2020-01", "2020-12")   # a whole AFRICA_REGIONS group
ni.index_since(root, dict_path, months_back=12)                  # relative window ending this month
```

Every result from `run_index`/`run_index_with_baseline` (and these shortcuts) carries its own config in `result.attrs["newsindex_config"]`, so `ni.rerun(result)` re-runs the exact same call -- handy for a monthly refresh script that just does `result = ni.rerun(last_result)`.

### Reading a result table

```python
print(ni.summarize(country))       # plain-English per-country summary
ni.print_period(country, "2020-01", "2020-06")  # pretty-print a slice
ni.describe(country)               # per-country min/max/mean/std + biggest move
ni.top(country, n=5, by="EPU")     # top-5 country-months
ni.report(country, "report.html")  # one-page HTML or Markdown report

ni.resample(country, freq="Q")               # monthly -> quarterly/annual averages
ni.rebase(country, ("2022-01", "2022-12"))   # recompute the 100-mean rescaling against a different period
ni.smooth(country, window=3)                 # adds an EPU_smoothed rolling-average column
```

### Diagnostics

Catch problems in an archive or dictionary before spending a long run on bad input:

```python
ni.coverage(root)                      # which (country, year-month) pairs have files, and how many
ni.validate_archive(root)              # missing columns / empty files / date-vs-folder mismatches
ni.dictionary_coverage(kw, root)       # countries in the archive with no dictionary entry, and vice versa
ni.health_check(root, dict_path)       # one-call "is everything wired up correctly" smoke test
```

### Weighting presets

Ready-made weight sources for `aggregate()`'s `weights` argument, for cases that would otherwise mean writing the same table or lambda by hand:

```python
ni.weights.equal                               # equal weighting -- pass directly, no call
ni.weights.gdp("gdp.csv")                      # same as aggregate.monthly_gdp()
ni.weights.population("population.csv")        # same interpolation, a population column instead
ni.weights.custom("share")                     # weight by an existing column of country_index
ni.weights.pca(indicators_table)                # composite weight from the first principal component
                                                 # across several indicator columns (e.g. gdp + population)
```

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

`read_articles()` here reads the whole requested scope into memory at once, which is simple and fine for a small archive (an example, a notebook, a few months of one country) but not for a large one -- see [Performance](#performance) for `index.count_archive()`, the memory-safe alternative `run_index`/`run_index_with_baseline` already use.

## Performance

A large archive (many countries, many newspapers, several years) can be too much article text to hold in memory at once, and classifying it can be slow if done naively. Two things address this, independent of each other:

- **`index.count_archive(root, dictionary, ...)`** goes straight from the archive root to the final per-newspaper-month counts, without ever building a table of the whole requested scope. It processes one (country, year-month) batch at a time -- every newspaper's articles for one month of one country -- counts it, and discards the raw text before moving to the next batch. Peak memory stays close to the size of one country-month's articles, regardless of how many countries, newspapers, or years `root` spans. `run_index()`/`run_index_with_baseline()` use this internally, so this is automatic for anyone using those. Pass a `progress` callback (they do, printing one line per batch via `verbose=True`) to see it moving through a long run instead of it looking stalled.
- **Classification is vectorized and short-circuits.** `index.count()` (and therefore `count_archive()`) checks each component (Economic, Policy, Uncertainty) with one combined regex across a whole batch at once, instead of looping over every dictionary term for every article in Python. It also checks a cheap, over-inclusive pattern first and only runs the precise, word-boundary-checked pattern on the rows that pass it (`match.loose_component_pattern()` / `match.component_pattern()`) -- and since EPU requires all three components, each one only re-checks the rows that survived the previous one. Most articles don't contain any term from a given component at all, so this avoids the expensive pattern almost entirely for them.

`read_articles()`'s `country`/`site`/`year_month` columns are also categorical (each distinct value stored once, not once per row), which matters when that table is large.

None of this changes any function's output -- `count_archive()` produces the same counts as `count(dedupe(read_articles(...)), dictionary)` (see its docstring for the one documented edge case around articles whose actual date disagrees with their month folder).

## Modules

| Module | Responsibility | Main functions |
| --- | --- | --- |
| `match` | Normalize text (Unicode NFKC, case-fold, quotes, dashes) and match whole words/phrases in any script | `normalize()`, `find_terms()`, `is_epu()` |
| `dictionary` | Load and validate country keyword files; enforce disjoint components; produce an audit and a fingerprint | `load()`, `validate()`, `fingerprint()` |
| `dedup` | Remove duplicate articles within a country-month by canonical URL, or by title/text when the URL is missing | `canonical_url()`, `dedupe()` |
| `index` | Count EPU articles per newspaper-month (from a table, or streamed straight from an archive); compute a single-window index, or fit a frozen baseline and score later windows against it | `count()`, `count_archive()`, `compute_index()`, `fit_baseline()`, `score()`, `score_windows()` |
| `aggregate` | Country z-scores, weighting (GDP-PPP by default, or your own table/function), regional and Africa series; annual-to-monthly GDP | `monthly_gdp()`, `aggregate()` |
| `weights` | Ready-made weight sources for `aggregate()` | `equal`, `gdp()`, `population()`, `custom()`, `pca()` |
| `diagnostics` | Catch archive/dictionary problems before a long run | `coverage()`, `validate_archive()`, `dictionary_coverage()`, `health_check()` |

`newsindex.read_articles(root)` reads the one input these functions don't take as a table: a directory of scraped article archives. `newsindex.run_index()`/`run_index_with_baseline()` (see Quick start above) compose the whole pipeline in one call, with named shortcuts (`quick_index`, `index_all`, `index_countries`, `index_region`, `index_since`, `rerun`) for common request shapes. `newsindex.summarize()`/`print_period()`/`describe()`/`top()`/`report()`/`resample()`/`rebase()`/`smooth()` (from the `report` module, re-exported at the package root) turn a result table into something readable without opening pandas. `newsindex.countries()` is a standalone reference table (see below), not part of the EPU pipeline itself.

## Countries

`newsindex.countries()` returns a global country reference table -- not just Africa, and not specific to this library's own naming -- for building your own country groupings (e.g. for `aggregate.aggregate`'s `groups` argument) without compiling a list by hand:

```python
>>> ni.countries().head(3)
          name iso2 iso3 continent    sub_region intermediate_region
0  Afghanistan   AF  AFG      Asia Southern Asia
1      Albania   AL  ALB    Europe Southern Europe
2      Algeria   DZ  DZA    Africa Northern Africa
```

All 249 ISO 3166-1 entries (UN member states and dependencies/territories alike) are included, each with its UN M49 geographic classification. That classification is three columns, and the third is only populated for some countries:

- `continent` -- one of 5 values.
- `sub_region` -- one of 17 values.
- `intermediate_region` -- one of 7 values, set **only** for Africa and the Americas (e.g. Nigeria's `sub_region` is "Sub-Saharan Africa" and its `intermediate_region` is "Western Africa"); every other continent's `sub_region` is already the fine-grained level, so `intermediate_region` is blank there.

The full hierarchy:

- **Africa**
  - Northern Africa
  - Sub-Saharan Africa (Eastern Africa, Middle Africa, Southern Africa, Western Africa)
- **Americas**
  - Latin America and the Caribbean (Caribbean, Central America, South America)
  - Northern America
- **Asia**
  - Central Asia
  - Eastern Asia
  - South-eastern Asia
  - Southern Asia
  - Western Asia
- **Europe**
  - Eastern Europe
  - Northern Europe
  - Southern Europe
  - Western Europe
- **Oceania**
  - Australia and New Zealand
  - Melanesia
  - Micronesia
  - Polynesia

`continent`/`sub_region`/`intermediate_region` are blank for the two ISO entries UN M49 does not classify (Antarctica; Taiwan, which is not a UN member).

`name` is ISO 3166-1's official short name, which does not always match the names used elsewhere in this library -- e.g. this table's "Congo, Democratic Republic of the" is `AFRICA_REGIONS`'s and the EPU dictionary's "Congo DRC". That's expected: `countries()` is a neutral standard reference, independent of this project's own naming, the same way a GDP source file's "Cabo Verde" needs `DEFAULT_COUNTRY_ALIASES` to line up with "Cape Verde" elsewhere.

`countries()` is purely a reference table: nothing else in this library reads it automatically. `AFRICA_REGIONS` (CAPE's own published Africa EPU grouping) and any `groups` you pass to `aggregate.aggregate` yourself are unaffected by it and unchanged.

For the full 249-row table, call `countries()` or see [`data/countries.csv`](data/countries.csv).

## Method

- **Classify.** An article counts as EPU if its title or body contains at least one Economic, one Policy, and one Uncertainty term from its country's dictionary. A term belongs to one component only.
- **Scale.** `S_it = EPU articles_it / all articles_it` for newspaper *i* in month *t*.
- **Standardise.** `N_it = S_it / sigma_i`, where `sigma_i` is the newspaper's population standard deviation over the baseline window. The window is always chosen by the caller (the published CAPE Africa EPU index uses 2015-01 to 2025-12; the library has no hardcoded default).
- **Average and normalise.** `M_t` is the mean of `N_it` across newspapers; `EPU_t = 100 * M_t / mu`, where `mu` is the baseline mean of `M_t`.
- **Freeze.** `sigma_i`, `mu`, and the retained newspapers are saved as a JSON baseline (`Baseline.save`/`Baseline.load`); later months are scored against it and never recalibrated.
- **Aggregate.** `z_ct = (EPU_ct - m_c) / s_c`; regional `Z_rt = sum(w_ct * z_ct)`, with weights `w_ct` GDP-PPP by default (or any table/function you supply, see below); rescaled to `100 + 10 * standardised Z_rt`. The Africa series applies the same steps across all countries directly — it is not an average of the regional series.

### Two ways to compute an index

- **No baseline (`compute_index`).** One window. `sigma`/`mu` are fit *from* that window and used to score that same window — nothing is frozen or reused. `ni.index.compute_index(counts, start="2020-01", end="2020-12")`.
- **With baseline (`fit_baseline` + `score` / `score_windows`).** Two or more windows. The first window is frozen once (`fit_baseline`); any later window (or several) is scored against that frozen baseline, never refitting it:
  ```python
  baseline = ni.index.fit_baseline(counts, start="2015-01", end="2024-12")
  later = ni.index.score_windows(counts, baseline, windows=[("2025-01", "2025-12")])
  ```
  `score_windows` just calls `score()` once per `(start, end)` pair and tags each result with a `window` column; `score(counts, baseline)` alone is equivalent when you only need one later window and don't need that column.

### Weighting regional aggregation

`aggregate.aggregate`'s `weights` argument isn't locked to GDP-PPP. It accepts either:

- **A table** — `country`/`year_month`/value columns, like `monthly_gdp()`'s output. Use `weight_col` if the value column isn't named `"gdp"` (e.g. a population or trade-volume table you built yourself). Only countries present in the table are included, and within each group-month they're weighted by their share of the group's total.
- **A function** — `weights(month_rows, group_name) -> pandas.Series`, called once per group per month with that month's rows (`country`, `z`, …) and the group's name; return a non-negative weight per row. Weights don't need to be pre-normalised, so equal weighting is just `lambda rows, group: pandas.Series(1.0, index=rows.index)`.

```python
population = pd.DataFrame({"country": [...], "year_month": [...], "population": [...]})
by_population = ni.aggregate.aggregate(country, population, weight_col="population")

equal_weighted = ni.aggregate.aggregate(country, lambda rows, group: pd.Series(1.0, index=rows.index))
```

`monthly_gdp()` itself also accepts an in-memory DataFrame in place of a file path (`source` argument), for a weight table that isn't on disk as annual-GDP-shaped CSV — any annual by-country metric, not just GDP, interpolates the same way.

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
