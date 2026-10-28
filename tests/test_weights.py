"""weights.equal / gdp / population / custom / pca."""

import numpy as np
import pandas as pd
import pytest

import newsindex as ni
from newsindex import aggregate as agg


def _country_index() -> pd.DataFrame:
    rows = []
    for country, base in [("Nigeria", 100.0), ("Kenya", 120.0), ("Ghana", 90.0)]:
        for i, month in enumerate(["2020-01", "2020-02", "2020-03", "2020-04"]):
            rows.append({"country": country, "year_month": month, "EPU": base + i * 5})
    return pd.DataFrame(rows)


def test_weights_equal_matches_the_callable_signature():
    month_rows = pd.DataFrame({"country": ["Nigeria", "Kenya"], "z": [0.5, -0.5]})

    result = ni.weights.equal(month_rows, "West")

    assert list(result) == [1.0, 1.0]
    assert list(result.index) == list(month_rows.index)


def test_weights_gdp_matches_monthly_gdp(tmp_path):
    source = pd.DataFrame(
        {"Country": ["Nigeria", "Nigeria"], "Year": [2019, 2020], "GDP_Billions_USD": [100.0, 110.0]}
    )

    from_weights = ni.weights.gdp(source)
    from_aggregate = agg.monthly_gdp(source)

    pd.testing.assert_frame_equal(from_weights, from_aggregate)


def test_weights_population_interpolates_like_monthly_gdp_but_with_its_own_column():
    source = pd.DataFrame(
        {"Country": ["Nigeria", "Nigeria"], "Year": [2019, 2020], "Population": [200.0, 210.0]}
    )

    result = ni.weights.population(source)

    assert "population" in result.columns
    assert len(result) == 24  # 2 years * 12 months
    assert result["population"].iloc[0] == pytest.approx(200.0)


def test_weights_custom_reads_an_existing_column_through_aggregate():
    country_index = _country_index()
    country_index["share"] = [1.0] * len(country_index)

    result = agg.aggregate(country_index, ni.weights.custom("share"), groups={"West": ["Nigeria", "Ghana"]})

    assert set(result["group"]) >= {"West"}


def test_weights_pca_returns_a_non_negative_weight_column():
    table = pd.DataFrame(
        {
            "country": ["Nigeria", "Kenya", "Ghana", "Nigeria", "Kenya", "Ghana"],
            "year_month": ["2020-01"] * 3 + ["2020-02"] * 3,
            "gdp": [500.0, 100.0, 70.0, 510.0, 101.0, 71.0],
            "population": [200.0, 50.0, 30.0, 201.0, 51.0, 31.0],
        }
    )

    result = ni.weights.pca(table)

    assert "pca_weight" in result.columns
    assert (result["pca_weight"] >= 0).all()
    assert list(result["country"]) == list(table["country"])


def test_weights_pca_raises_with_fewer_than_two_indicator_columns():
    table = pd.DataFrame({"country": ["Nigeria", "Kenya", "Ghana"], "year_month": ["2020-01"] * 3, "gdp": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="at least 2"):
        ni.weights.pca(table)


def test_weights_pca_raises_with_fewer_than_three_rows():
    table = pd.DataFrame(
        {"country": ["Nigeria", "Kenya"], "year_month": ["2020-01"] * 2, "gdp": [1.0, 2.0], "population": [3.0, 4.0]}
    )

    with pytest.raises(ValueError, match="at least 3 rows"):
        ni.weights.pca(table)


def test_weights_pca_verbose_prints_loadings(capsys):
    table = pd.DataFrame(
        {
            "country": ["Nigeria", "Kenya", "Ghana"],
            "year_month": ["2020-01"] * 3,
            "gdp": [500.0, 100.0, 70.0],
            "population": [200.0, 50.0, 30.0],
        }
    )

    ni.weights.pca(table, verbose=True)

    assert "PCA weights" in capsys.readouterr().err
