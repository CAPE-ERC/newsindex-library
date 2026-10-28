"""summarize / print_period / describe / top / report: reading a result table."""

import gzip
from pathlib import Path

import pandas as pd
import pytest

import newsindex as ni

DATA_DICTIONARY = Path(__file__).resolve().parents[1] / "data" / "eptu_country_keywords.json"


def _write(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "site,date,title,url,text\n" + "\n".join(rows)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(body)


def _build_archive(root: Path) -> None:
    for month, epu_count, plain_count in [("2020-01", 1, 3), ("2020-02", 2, 2), ("2020-03", 1, 3)]:
        rows = [
            f'The Punch,{month}-01,"Government fiscal policy and inflation create uncertainty",'
            f"https://example.com/nigeria/{month}/{i},body"
            for i in range(epu_count)
        ] + [
            f'The Punch,{month}-02,"Local football result {i}",https://example.com/nigeria/{month}/p{i},body'
            for i in range(plain_count)
        ]
        _write(root / "Nigeria" / "The Punch" / month / f"articles_The Punch_{month}.csv.gz", rows)

    for month, epu_count, plain_count in [("2020-01", 2, 2), ("2020-02", 1, 3), ("2020-03", 2, 2)]:
        rows = [
            f'Daily Nation,{month}-01,"Fiscal policy and inflation create unstable economy",'
            f"https://example.com/kenya/{month}/{i},body"
            for i in range(epu_count)
        ] + [
            f'Daily Nation,{month}-02,"Weather update {i}",https://example.com/kenya/{month}/p{i},body'
            for i in range(plain_count)
        ]
        _write(root / "Kenya" / "Daily Nation" / month / f"articles_Daily Nation_{month}.csv.gz", rows)


@pytest.fixture
def result(tmp_path) -> pd.DataFrame:
    _build_archive(tmp_path / "archive")
    return ni.run_index(tmp_path / "archive", DATA_DICTIONARY, "2020-01", "2020-03", verbose=False)


def test_summarize_mentions_every_country_with_first_last_and_pct_change(result):
    text = ni.summarize(result)

    assert "Nigeria" in text and "Kenya" in text
    assert "2020-01" in text and "2020-03" in text
    assert "%" in text


def test_summarize_single_month_country_has_no_pct_change_line():
    single = pd.DataFrame(
        {"country": ["Nigeria"], "year_month": ["2020-01"], "EPU": [50.0], "M_t": [1.0], "mu": [1.0], "n_sites": [1]}
    )
    text = ni.summarize(single)
    assert "only month available" in text
    assert "%" not in text


def test_print_period_returns_the_filtered_window_and_prints(result, capsys):
    filtered = ni.print_period(result, "2020-01", "2020-01")

    assert set(filtered["year_month"]) == {"2020-01"}
    assert "2020-01" in capsys.readouterr().out


def test_print_period_raises_on_an_empty_window(result):
    with pytest.raises(ValueError, match="no rows"):
        ni.print_period(result, "2099-01", "2099-12")


def test_describe_has_one_row_per_country_with_biggest_move(result):
    stats = ni.describe(result)

    assert set(stats["country"]) == {"Nigeria", "Kenya"}
    assert (stats["n_months"] == 3).all()
    assert stats["biggest_move"].notna().all()
    assert stats["biggest_move_month"].notna().all()


def test_describe_single_month_country_has_nan_biggest_move():
    single = pd.DataFrame(
        {"country": ["Nigeria"], "year_month": ["2020-01"], "EPU": [50.0], "M_t": [1.0], "mu": [1.0], "n_sites": [1]}
    )
    stats = ni.describe(single)

    assert stats.loc[0, "n_months"] == 1
    assert pd.isna(stats.loc[0, "biggest_move"])
    assert stats.loc[0, "biggest_move_month"] is None


def test_top_sorts_descending_by_default_column(result):
    top3 = ni.top(result, n=3)

    assert len(top3) == 3
    assert list(top3["EPU"]) == sorted(top3["EPU"], reverse=True)


def test_top_raises_a_clear_error_for_an_unknown_column(result):
    with pytest.raises(ValueError, match="nope"):
        ni.top(result, by="nope")


def test_report_html(tmp_path, result):
    out = tmp_path / "report.html"
    ni.report(result, out)

    content = out.read_text(encoding="utf-8")
    assert "<html>" in content
    assert "Nigeria" in content


def test_report_markdown(tmp_path, result):
    out = tmp_path / "report.md"
    ni.report(result, out)

    content = out.read_text(encoding="utf-8")
    assert content.startswith("# newsindex EPU report")
    assert "| country |" in content


def test_report_rejects_an_unsupported_extension(tmp_path, result):
    with pytest.raises(ValueError, match="unsupported"):
        ni.report(result, tmp_path / "report.pdf")


def test_baseline_repr_is_readable():
    baseline = ni.index.Baseline(
        start="2015-01",
        end="2024-12",
        mu={"Nigeria": 1.0, "Kenya": 1.0, "Ghana": 1.0},
        sigma={"Nigeria": {"a": 0.1, "b": 0.2}, "Kenya": {"c": 0.1}, "Ghana": {"d": 0.1}},
    )
    text = repr(baseline)

    assert text.startswith("Baseline(2015-01..2024-12, 3 countries")
    assert "Ghana" in text and "Kenya" in text and "Nigeria" in text


def test_baseline_repr_truncates_past_five_countries():
    mu = {f"Country{i}": 1.0 for i in range(8)}
    sigma = {name: {"site": 0.1} for name in mu}
    baseline = ni.index.Baseline(start="2015-01", end="2024-12", mu=mu, sigma=sigma)

    assert "+3 more" in repr(baseline)


def test_resample_quarterly_averages_within_each_quarter(result):
    quarterly = ni.resample(result, freq="Q")

    assert set(quarterly["period"]) == {"2020Q1"}
    assert set(quarterly["country"]) == {"Nigeria", "Kenya"}


def test_resample_annual_accepts_a_and_y():
    monthly = pd.DataFrame(
        {"country": ["Nigeria"] * 3, "year_month": ["2020-01", "2020-02", "2020-03"], "EPU": [90.0, 100.0, 110.0]}
    )

    annual_a = ni.resample(monthly, freq="A")
    annual_y = ni.resample(monthly, freq="Y")

    assert list(annual_a["period"]) == ["2020"]
    assert annual_a["EPU"].iloc[0] == pytest.approx(100.0)
    pd.testing.assert_frame_equal(annual_a, annual_y)


def test_resample_raises_on_an_unsupported_freq(result):
    with pytest.raises(ValueError, match="unsupported freq"):
        ni.resample(result, freq="W")


def test_rebase_rescales_each_country_to_a_new_reference_period():
    series = pd.DataFrame(
        {
            "country": ["Nigeria"] * 2,
            "year_month": ["2020-01", "2020-02"],
            "EPU": [50.0, 100.0],
        }
    )

    rebased = ni.rebase(series, ("2020-01", "2020-01"))

    assert rebased.loc[rebased["year_month"] == "2020-01", "EPU"].iloc[0] == pytest.approx(100.0)
    assert rebased.loc[rebased["year_month"] == "2020-02", "EPU"].iloc[0] == pytest.approx(200.0)


def test_rebase_raises_naming_a_country_with_no_rows_in_the_new_period():
    series = pd.DataFrame({"country": ["Nigeria"], "year_month": ["2020-01"], "EPU": [50.0]})

    with pytest.raises(ValueError, match="Nigeria"):
        ni.rebase(series, ("2099-01", "2099-12"))


def test_rebase_raises_on_a_zero_mean_baseline_period():
    series = pd.DataFrame({"country": ["Nigeria"], "year_month": ["2020-01"], "EPU": [0.0]})

    with pytest.raises(ValueError, match="zero mean"):
        ni.rebase(series, ("2020-01", "2020-01"))


def test_smooth_adds_a_rolling_column_without_touching_raw_epu():
    series = pd.DataFrame(
        {
            "country": ["Nigeria"] * 4,
            "year_month": ["2020-01", "2020-02", "2020-03", "2020-04"],
            "EPU": [10.0, 20.0, 30.0, 40.0],
        }
    )

    smoothed = ni.smooth(series, window=2)

    assert list(smoothed["EPU"]) == [10.0, 20.0, 30.0, 40.0]
    assert smoothed["EPU_smoothed"].iloc[0] == pytest.approx(10.0)
    assert smoothed["EPU_smoothed"].iloc[1] == pytest.approx(15.0)
    assert smoothed["EPU_smoothed"].iloc[3] == pytest.approx(35.0)


def test_smooth_raises_on_a_non_positive_window():
    series = pd.DataFrame({"country": ["Nigeria"], "year_month": ["2020-01"], "EPU": [10.0]})

    with pytest.raises(ValueError, match="window"):
        ni.smooth(series, window=0)
