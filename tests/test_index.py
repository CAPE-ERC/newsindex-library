import math

import pandas as pd
import pytest

from newsindex import dictionary, index


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
