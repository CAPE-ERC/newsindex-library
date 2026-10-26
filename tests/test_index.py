import gzip
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from newsindex import dedup, dictionary, index, read_articles


def test_count_counts_epu_and_total_articles_per_site_month(tmp_path):
    import json

    dict_path = tmp_path / "dict.json"
    dict_path.write_text(
        json.dumps(
            {
                "countries": {
                    "Freedonia": {
                        "languages": ["en"],
                        "ECONOMY_TERMS": ["economy"],
                        "POLICY_TERMS": ["government"],
                        "UNCERTAINTY_TERMS": ["uncertainty"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    kw = dictionary.load(dict_path)

    articles = pd.DataFrame(
        {
            "country": ["Freedonia", "Freedonia", "Freedonia"],
            "site": ["SiteA", "SiteA", "SiteA"],
            "year_month": ["2020-01", "2020-01", "2020-01"],
            "title": [
                "Government economy uncertainty rises",
                "Economy grows steadily",
                "Sports results this week",
            ],
            "text": ["", "", ""],
        }
    )
    counts = index.count(articles, kw)
    row = counts.iloc[0]
    assert row["epu_articles"] == 1
    assert row["all_articles"] == 3


def test_count_raises_on_country_missing_from_dictionary(tmp_path):
    import json

    dict_path = tmp_path / "dict.json"
    dict_path.write_text(
        json.dumps({"countries": {"Freedonia": {"languages": ["en"], "ECONOMY_TERMS": ["economy"], "POLICY_TERMS": ["government"], "UNCERTAINTY_TERMS": ["uncertainty"]}}}),
        encoding="utf-8",
    )
    kw = dictionary.load(dict_path)
    articles = pd.DataFrame(
        {
            "country": ["Sylvania"],
            "site": ["SiteA"],
            "year_month": ["2020-01"],
            "title": ["x"],
            "text": ["y"],
        }
    )
    with pytest.raises(KeyError):
        index.count(articles, kw)


def _hand_computed_counts() -> pd.DataFrame:
    rows = []
    # SiteA: S_it alternates 0.1, 0.3 -> population std = 0.1, retained.
    for month, epu, all_ in zip(
        ["2020-01", "2020-02", "2020-03", "2020-04"], [1, 3, 1, 3], [10, 10, 10, 10]
    ):
        rows.append({"country": "Freedonia", "site": "SiteA", "year_month": month, "epu_articles": epu, "all_articles": all_})
    # SiteB: constant S_it = 0.25 -> zero variance, dropped.
    for month in ["2020-01", "2020-02", "2020-03", "2020-04"]:
        rows.append({"country": "Freedonia", "site": "SiteB", "year_month": month, "epu_articles": 2, "all_articles": 8})
    # SiteC: a single baseline month -> too few observations, dropped.
    rows.append({"country": "Freedonia", "site": "SiteC", "year_month": "2020-01", "epu_articles": 1, "all_articles": 4})
    # SiteA continues into a post-baseline month.
    rows.append({"country": "Freedonia", "site": "SiteA", "year_month": "2020-05", "epu_articles": 3, "all_articles": 10})
    # SiteB also has a post-baseline month; it must not affect scoring since it was dropped.
    rows.append({"country": "Freedonia", "site": "SiteB", "year_month": "2020-05", "epu_articles": 4, "all_articles": 8})
    return pd.DataFrame(rows)


def test_fit_baseline_drops_zero_variance_and_single_observation_sites():
    counts = _hand_computed_counts()
    baseline = index.fit_baseline(counts, "2020-01", "2020-04")

    assert baseline.retained_sites("Freedonia") == ["SiteA"]
    assert baseline.sigma["Freedonia"]["SiteA"] == pytest.approx(0.1)
    # mu = mean(N_it) = mean([1, 3, 1, 3]) = 2.0
    assert baseline.mu["Freedonia"] == pytest.approx(2.0)


def test_score_matches_hand_computed_epu_and_freezes_dropped_sites():
    counts = _hand_computed_counts()
    baseline = index.fit_baseline(counts, "2020-01", "2020-04")
    scored = index.score(counts, baseline).set_index("year_month")

    # EPU_t = 100 * M_t / mu; M_t = N_it of the single retained site, SiteA.
    assert scored.loc["2020-01", "EPU"] == pytest.approx(50.0)
    assert scored.loc["2020-02", "EPU"] == pytest.approx(150.0)
    assert scored.loc["2020-01", "n_sites"] == 1

    # Post-baseline month: SiteA S=0.3 -> N=3 -> EPU=150; SiteB is ignored
    # entirely (dropped at fit_baseline time) even though it has 2020-05 data.
    assert scored.loc["2020-05", "EPU"] == pytest.approx(150.0)
    assert scored.loc["2020-05", "n_sites"] == 1


def test_compute_index_is_in_sample_fit_and_score_for_one_window():
    counts = _hand_computed_counts()
    result = index.compute_index(counts, "2020-01", "2020-04").set_index("year_month")

    # Same numbers as fit_baseline+score over the identical window: sigma and
    # mu are fit from (and only from) this one window, then it is scored
    # against itself -- nothing is frozen or reused.
    assert result.loc["2020-01", "EPU"] == pytest.approx(50.0)
    assert result.loc["2020-02", "EPU"] == pytest.approx(150.0)
    # The post-baseline 2020-05 row exists in counts but is outside this
    # single window, so it must not appear in the result.
    assert "2020-05" not in result.index


def test_score_windows_scores_multiple_named_periods_against_one_frozen_baseline():
    counts = pd.concat(
        [
            _hand_computed_counts(),
            pd.DataFrame(
                [{"country": "Freedonia", "site": "SiteA", "year_month": "2020-06", "epu_articles": 1, "all_articles": 10}]
            ),
        ],
        ignore_index=True,
    )
    baseline = index.fit_baseline(counts, "2020-01", "2020-04")

    result = index.score_windows(counts, baseline, windows=[("2020-05", "2020-05"), ("2020-06", "2020-06")])
    result = result.set_index(["window", "year_month"])

    assert result.loc[("2020-05:2020-05", "2020-05"), "EPU"] == pytest.approx(150.0)
    assert result.loc[("2020-06:2020-06", "2020-06"), "EPU"] == pytest.approx(50.0)
    assert set(result.index.get_level_values("window")) == {"2020-05:2020-05", "2020-06:2020-06"}


def test_score_windows_with_no_windows_returns_empty_frame_with_expected_columns():
    counts = _hand_computed_counts()
    baseline = index.fit_baseline(counts, "2020-01", "2020-04")
    result = index.score_windows(counts, baseline, windows=[])
    assert list(result.columns) == ["window", "country", "year_month", "EPU", "M_t", "mu", "n_sites"]
    assert len(result) == 0


def test_baseline_save_and_load_round_trip(tmp_path):
    counts = _hand_computed_counts()
    baseline = index.fit_baseline(counts, "2020-01", "2020-04", dictionary_fingerprint="abc123")
    path = tmp_path / "baseline.json"
    baseline.save(path)
    reloaded = index.Baseline.load(path)

    assert reloaded.mu == baseline.mu
    assert reloaded.sigma == baseline.sigma
    assert reloaded.dictionary_fingerprint == "abc123"
    assert math.isclose(reloaded.mu["Freedonia"], 2.0)


def _write_count_archive_dictionary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "countries": {
                    "Nigeria": {
                        "languages": ["en"],
                        "ECONOMY_TERMS": ["fiscal", "inflation"],
                        "POLICY_TERMS": ["government"],
                        "UNCERTAINTY_TERMS": ["uncertainty"],
                    },
                    "Kenya": {
                        "languages": ["en"],
                        "ECONOMY_TERMS": ["fiscal", "inflation"],
                        "POLICY_TERMS": ["government"],
                        "UNCERTAINTY_TERMS": ["uncertain"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _write_count_archive_articles(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "site,date,title,url,text\n" + "\n".join(rows)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(body)


def _build_count_archive(root: Path) -> None:
    _write_count_archive_articles(
        root / "Nigeria" / "The Punch" / "2020-01" / "articles_The Punch_2020-01.csv.gz",
        [
            'The Punch,2020-01-01,"Government fiscal policy creates uncertainty",https://x/n1,body',
            'The Punch,2020-01-02,"Government fiscal policy creates uncertainty",https://x/n1,body',  # duplicate URL
            "The Punch,2020-01-03,Local football result,https://x/n2,body",
        ],
    )
    _write_count_archive_articles(
        root / "Nigeria" / "Vanguard" / "2020-01" / "articles_Vanguard_2020-01.csv.gz",
        ['Vanguard,2020-01-04,"Inflation and government uncertainty rise",https://x/n3,body'],
    )
    _write_count_archive_articles(
        root / "Nigeria" / "The Punch" / "2020-02" / "articles_The Punch_2020-02.csv.gz",
        ["The Punch,2020-02-01,Weather update,https://x/n4,body"],
    )
    _write_count_archive_articles(
        root / "Kenya" / "Daily Nation" / "2020-01" / "articles_Daily Nation_2020-01.csv.gz",
        ['Daily Nation,2020-01-05,"Government fiscal plan uncertain economy",https://x/k1,body'],
    )


def test_count_archive_matches_count_dedupe_read_articles(tmp_path):
    _build_count_archive(tmp_path / "archive")
    dict_path = tmp_path / "dict.json"
    _write_count_archive_dictionary(dict_path)
    kw = dictionary.load(dict_path)

    streamed = index.count_archive(tmp_path / "archive", kw)
    whole = index.count(dedup.dedupe(read_articles(tmp_path / "archive")), kw)

    sort_cols = ["country", "site", "year_month"]
    pd.testing.assert_frame_equal(
        streamed.sort_values(sort_cols).reset_index(drop=True),
        whole.sort_values(sort_cols).reset_index(drop=True),
    )


def test_count_archive_dedupe_false_counts_duplicates_separately(tmp_path):
    _build_count_archive(tmp_path / "archive")
    dict_path = tmp_path / "dict.json"
    _write_count_archive_dictionary(dict_path)
    kw = dictionary.load(dict_path)

    deduped = index.count_archive(tmp_path / "archive", kw, dedupe=True)
    raw = index.count_archive(tmp_path / "archive", kw, dedupe=False)

    punch_deduped = deduped[(deduped["country"] == "Nigeria") & (deduped["site"] == "The Punch")]
    punch_raw = raw[(raw["country"] == "Nigeria") & (raw["site"] == "The Punch")]
    assert int(punch_raw["all_articles"].sum()) == int(punch_deduped["all_articles"].sum()) + 1


def test_count_archive_countries_filters(tmp_path):
    _build_count_archive(tmp_path / "archive")
    dict_path = tmp_path / "dict.json"
    _write_count_archive_dictionary(dict_path)
    kw = dictionary.load(dict_path)

    result = index.count_archive(tmp_path / "archive", kw, countries=["nigeria"])

    assert set(result["country"]) == {"Nigeria"}


def test_count_archive_calls_progress_once_per_batch(tmp_path):
    _build_count_archive(tmp_path / "archive")
    dict_path = tmp_path / "dict.json"
    _write_count_archive_dictionary(dict_path)
    kw = dictionary.load(dict_path)

    messages: list[str] = []
    index.count_archive(tmp_path / "archive", kw, progress=messages.append)

    # Nigeria/2020-01 (2 newspapers), Nigeria/2020-02, Kenya/2020-01
    assert len(messages) == 3
    assert any("Nigeria" in m and "2020-01" in m for m in messages)
