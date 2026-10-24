import math

import pandas as pd
import pytest

from newsindex import aggregate


def test_monthly_gdp_log_linear_interpolation_and_flat_last_year(tmp_path):
    csv_path = tmp_path / "gdp.csv"
    csv_path.write_text(
        "Country,Year,GDP_Billions_USD\n"
        "Freedonia,2020,100\n"
        "Freedonia,2021,200\n",
        encoding="utf-8",
    )
    monthly = aggregate.monthly_gdp(csv_path).set_index("year_month")

    assert monthly.loc["2020-01", "gdp"] == pytest.approx(100.0)
    # Midyear (July, month index 6) is the geometric mean of the two years.
    assert monthly.loc["2020-07", "gdp"] == pytest.approx(math.sqrt(100 * 200))
    # The last year on file has no successor: its months are held flat.
    assert monthly.loc["2021-01", "gdp"] == pytest.approx(200.0)
    assert monthly.loc["2021-12", "gdp"] == pytest.approx(200.0)


def test_monthly_gdp_carries_forward_past_end():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "gdp.csv"
        csv_path.write_text("Country,Year,GDP_Billions_USD\nFreedonia,2020,100\n", encoding="utf-8")
        monthly = aggregate.monthly_gdp(csv_path, end="2022-06").set_index("year_month")
    assert monthly.loc["2021-06", "gdp"] == pytest.approx(100.0)
    assert monthly.loc["2022-06", "gdp"] == pytest.approx(100.0)
    assert "2022-07" not in monthly.index


def test_monthly_gdp_applies_default_country_aliases(tmp_path):
    csv_path = tmp_path / "gdp.csv"
    csv_path.write_text(
        "Country,Year,GDP_Billions_USD\nCabo Verde,2020,10\nBurkina Faso,2020,5\n", encoding="utf-8"
    )
    monthly = aggregate.monthly_gdp(csv_path)
    assert set(monthly["country"]) == {"Cape Verde", "Burkina-Faso"}


def _country_index() -> pd.DataFrame:
    months = ["2020-01", "2020-02", "2020-03", "2020-04"]
    return pd.DataFrame(
        {
            "country": ["X"] * 4 + ["Y"] * 4,
            "year_month": months + months,
            "EPU": [100, 110, 90, 100] + [100, 90, 110, 100],
        }
    )


def _gdp() -> pd.DataFrame:
    months = ["2020-01", "2020-02", "2020-03", "2020-04"]
    return pd.DataFrame(
        {
            "country": ["X"] * 4 + ["Y"] * 4,
            "year_month": months + months,
            "gdp": [60.0] * 4 + [40.0] * 4,
        }
    )


def test_aggregate_matches_hand_computed_weighted_zscore():
    groups = {"TestRegion": ["X", "Y"]}
    result = aggregate.aggregate(
        _country_index(), _gdp(), groups, baseline_start="2020-01", baseline_end="2020-04"
    )
    region = result[result["group"] == "TestRegion"].set_index("year_month")

    # z_X and z_Y both have baseline std sqrt(50); weighted z = 0.2 * z_X,t
    # (since z_Y,t = -z_X,t here), so the weighted series is
    # [0, 0.2*sqrt(2), -0.2*sqrt(2), 0] with population std 0.2.
    assert region.loc["2020-01", "z"] == pytest.approx(0.0, abs=1e-9)
    assert region.loc["2020-02", "z"] == pytest.approx(0.2 * math.sqrt(2))
    assert region.loc["2020-03", "z"] == pytest.approx(-0.2 * math.sqrt(2))

    assert region.loc["2020-01", "index"] == pytest.approx(100.0)
    assert region.loc["2020-02", "index"] == pytest.approx(100.0 + 50.0 * 0.2 * math.sqrt(2))
    assert region.loc["2020-02", "n_countries"] == 2
    assert region.loc["2020-02", "gdp_covered"] == pytest.approx(100.0)


def test_aggregate_africa_group_defaults_to_union_of_all_regions():
    groups = {"TestRegion": ["X", "Y"]}
    result = aggregate.aggregate(
        _country_index(), _gdp(), groups, baseline_start="2020-01", baseline_end="2020-04"
    )
    assert "Africa" in set(result["group"])
    africa = result[result["group"] == "Africa"].set_index("year_month")
    region = result[result["group"] == "TestRegion"].set_index("year_month")
    pd.testing.assert_series_equal(africa["index"], region["index"], check_names=False)


def test_aggregate_can_disable_africa_group():
    groups = {"TestRegion": ["X", "Y"]}
    result = aggregate.aggregate(
        _country_index(),
        _gdp(),
        groups,
        baseline_start="2020-01",
        baseline_end="2020-04",
        include_africa=False,
    )
    assert "Africa" not in set(result["group"])
