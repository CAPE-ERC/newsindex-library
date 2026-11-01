"""run_index / run_index_with_baseline: one-call orchestration."""

import datetime
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


def test_run_index_computes_a_single_window_index(tmp_path, capsys):
    _build_archive(tmp_path / "archive")

    result = ni.run_index(tmp_path / "archive", DATA_DICTIONARY, "2020-01", "2020-03")

    assert set(result["country"]) <= {"Nigeria", "Kenya"}
    assert (result["EPU"] >= 0).all()
    assert capsys.readouterr().err  # progress printed by default


def test_run_index_verbose_false_is_silent(tmp_path, capsys):
    _build_archive(tmp_path / "archive")

    ni.run_index(tmp_path / "archive", DATA_DICTIONARY, "2020-01", "2020-03", verbose=False)

    assert capsys.readouterr().err == ""


def test_run_index_only_streams_the_requested_window_not_earlier_archive_history(tmp_path, capsys):
    archive = tmp_path / "archive"
    _build_archive(archive)
    # A month well outside the requested 2020-01..2020-03 window -- run_index
    # must never read it, or mention it in progress output.
    _write(
        archive / "Nigeria" / "The Punch" / "2015-06" / "articles_The Punch_2015-06.csv.gz",
        ['The Punch,2015-06-01,"Old fiscal policy uncertainty story",https://example.com/old,body'],
    )

    ni.run_index(archive, DATA_DICTIONARY, "2020-01", "2020-03")

    progress_output = capsys.readouterr().err
    assert "2015-06" not in progress_output


def test_run_index_countries_filters_the_archive(tmp_path):
    _build_archive(tmp_path / "archive")

    result = ni.run_index(
        tmp_path / "archive", DATA_DICTIONARY, "2020-01", "2020-03", countries=["Nigeria"], verbose=False
    )

    assert set(result["country"]) == {"Nigeria"}


def test_run_index_raises_a_clear_error_for_a_missing_dictionary(tmp_path):
    _build_archive(tmp_path / "archive")

    with pytest.raises(FileNotFoundError, match="no-such-file.json"):
        ni.run_index(
            tmp_path / "archive", tmp_path / "no-such-file.json", "2020-01", "2020-03", verbose=False
        )


def test_run_index_with_baseline_fits_and_scores_in_one_call(tmp_path):
    _build_archive(tmp_path / "archive")

    result, baseline = ni.run_index_with_baseline(
        tmp_path / "archive",
        DATA_DICTIONARY,
        windows=[("2020-03", "2020-03")],
        baseline_start="2020-01",
        baseline_end="2020-02",
        verbose=False,
    )

    assert baseline.start == "2020-01" and baseline.end == "2020-02"
    assert set(result["window"]) == {"2020-03:2020-03"}
    assert set(result["country"]) <= {"Nigeria", "Kenya"}


def test_run_index_with_baseline_accepts_an_already_fitted_baseline(tmp_path):
    archive = tmp_path / "archive"
    _build_archive(archive)
    kw = ni.dictionary.load(DATA_DICTIONARY)
    counts = ni.index.count(ni.dedup.dedupe(ni.read_articles(archive)), kw)
    fitted = ni.index.fit_baseline(counts, start="2020-01", end="2020-02")

    result, returned_baseline = ni.run_index_with_baseline(
        archive, DATA_DICTIONARY, windows=[("2020-03", "2020-03")], baseline=fitted, verbose=False
    )

    assert returned_baseline is fitted
    assert set(result["window"]) == {"2020-03:2020-03"}


def test_run_index_with_baseline_requires_a_baseline_or_a_window(tmp_path):
    _build_archive(tmp_path / "archive")

    with pytest.raises(ValueError, match="baseline"):
        ni.run_index_with_baseline(
            tmp_path / "archive", DATA_DICTIONARY, windows=[("2020-03", "2020-03")], verbose=False
        )


def test_run_index_with_baseline_rejects_conflicting_baseline_arguments(tmp_path):
    archive = tmp_path / "archive"
    _build_archive(archive)
    kw = ni.dictionary.load(DATA_DICTIONARY)
    counts = ni.index.count(ni.dedup.dedupe(ni.read_articles(archive)), kw)
    fitted = ni.index.fit_baseline(counts, start="2020-01", end="2020-02")

    with pytest.raises(ValueError, match="baseline_start"):
        ni.run_index_with_baseline(
            archive,
            DATA_DICTIONARY,
            windows=[("2020-03", "2020-03")],
            baseline=fitted,
            baseline_start="2020-01",
            verbose=False,
        )

def test_quick_index_resolves_one_country_and_one_calendar_year(tmp_path):
    _build_archive(tmp_path / "archive")

    result = ni.quick_index(tmp_path / "archive", DATA_DICTIONARY, "Nigeria", 2020, verbose=False)

    assert set(result["country"]) == {"Nigeria"}
    assert set(result["year_month"]) <= {"2020-01", "2020-02", "2020-03"}


def test_index_all_covers_every_country_folder(tmp_path):
    _build_archive(tmp_path / "archive")

    result = ni.index_all(tmp_path / "archive", DATA_DICTIONARY, "2020-01", "2020-03", verbose=False)

    assert set(result["country"]) == {"Nigeria", "Kenya"}


def test_index_countries_filters_to_the_given_list(tmp_path):
    _build_archive(tmp_path / "archive")

    result = ni.index_countries(
        tmp_path / "archive", DATA_DICTIONARY, ["Kenya"], "2020-01", "2020-03", verbose=False
    )

    assert set(result["country"]) == {"Kenya"}


def test_index_region_resolves_a_named_region_present_in_the_archive(tmp_path):
    _build_archive(tmp_path / "archive")

    result = ni.index_region(
        tmp_path / "archive", DATA_DICTIONARY, "West", "2020-01", "2020-03", verbose=False
    )

    assert set(result["country"]) == {"Nigeria"}


def test_index_region_accepts_an_explicit_country_list(tmp_path):
    _build_archive(tmp_path / "archive")

    result = ni.index_region(
        tmp_path / "archive", DATA_DICTIONARY, ["Nigeria", "Kenya"], "2020-01", "2020-03", verbose=False
    )

    assert set(result["country"]) == {"Nigeria", "Kenya"}


def test_index_region_raises_for_an_unknown_region_name(tmp_path):
    _build_archive(tmp_path / "archive")

    with pytest.raises(ValueError, match="unknown region"):
        ni.index_region(
            tmp_path / "archive", DATA_DICTIONARY, "NoSuchRegion", "2020-01", "2020-03", verbose=False
        )


def test_index_region_raises_when_no_member_folder_exists(tmp_path):
    _build_archive(tmp_path / "archive")

    with pytest.raises(ValueError, match="none of region"):
        ni.index_region(tmp_path / "archive", DATA_DICTIONARY, "North", "2020-01", "2020-03", verbose=False)


def test_index_since_covers_the_expected_relative_window(tmp_path, monkeypatch):
    _build_archive(tmp_path / "archive")
    real_date = datetime.date
    fixed_date = type("FixedDate", (), {"today": staticmethod(lambda: real_date(2020, 3, 15))})
    monkeypatch.setattr(ni.pipeline.datetime, "date", fixed_date)

    result = ni.index_since(tmp_path / "archive", DATA_DICTIONARY, months_back=3, verbose=False)

    assert set(result["year_month"]) <= {"2020-01", "2020-02", "2020-03"}


def test_rerun_reproduces_a_run_index_result(tmp_path):
    _build_archive(tmp_path / "archive")

    first = ni.run_index(tmp_path / "archive", DATA_DICTIONARY, "2020-01", "2020-03", verbose=False)
    again = ni.rerun(first)

    pd.testing.assert_frame_equal(first, again)


def test_rerun_reproduces_a_run_index_with_baseline_result_fit_fresh(tmp_path):
    _build_archive(tmp_path / "archive")

    first, _ = ni.run_index_with_baseline(
        tmp_path / "archive",
        DATA_DICTIONARY,
        windows=[("2020-03", "2020-03")],
        baseline_start="2020-01",
        baseline_end="2020-02",
        verbose=False,
    )
    again, _ = ni.rerun(first)

    pd.testing.assert_frame_equal(first, again)


def test_rerun_reproduces_a_run_index_with_baseline_result_reusing_a_baseline(tmp_path):
    archive = tmp_path / "archive"
    _build_archive(archive)
    kw = ni.dictionary.load(DATA_DICTIONARY)
    counts = ni.index.count(ni.dedup.dedupe(ni.read_articles(archive)), kw)
    fitted = ni.index.fit_baseline(counts, start="2020-01", end="2020-02")

    first, _ = ni.run_index_with_baseline(
        archive, DATA_DICTIONARY, windows=[("2020-03", "2020-03")], baseline=fitted, verbose=False
    )
    again, returned_baseline = ni.rerun(first)

    assert returned_baseline is fitted
    pd.testing.assert_frame_equal(first, again)


def test_rerun_raises_a_clear_error_without_a_stored_config(tmp_path):
    archive = tmp_path / "archive"
    _build_archive(archive)
    kw = ni.dictionary.load(DATA_DICTIONARY)
    counts = ni.index.count(ni.dedup.dedupe(ni.read_articles(archive)), kw)
    plain_result = ni.index.compute_index(counts, "2020-01", "2020-03")

    with pytest.raises(ValueError, match="rerun"):
        ni.rerun(plain_result)

@pytest.mark.parametrize("reuse_baseline", [False, True], ids=["fresh", "reused"])
def test_run_index_with_baseline_reads_only_required_months(
    tmp_path, monkeypatch, reuse_baseline
):
    """Check that baseline scoring skips irrelevant archive months."""
    archive = tmp_path / "archive"
    _build_archive(archive)

    # Add two months outside the baseline and scoring windows.
    for month in ("2015-06", "2024-06"):
        _write(
            archive / "Nigeria" / "The Punch" / month
            / f"articles_The Punch_{month}.csv.gz",
            [
                f'The Punch,{month}-01,"Old fiscal policy uncertainty story",'
                f'https://example.com/nigeria/{month}/old,body'
            ],
        )

    baseline = None
    if reuse_baseline:
        keywords = ni.dictionary.load(DATA_DICTIONARY)
        articles = ni.read_articles(archive)
        counts = ni.index.count(ni.dedup.dedupe(articles), keywords)
        baseline = ni.index.fit_baseline(
            counts, start="2020-01", end="2020-02"
        )

    # Record every monthly article file opened.
    import newsindex.articles as article_module

    opened_months = []
    original_read_one = article_module._read_one

    def recording_read_one(file, country, encoding):
        opened_months.append(file.parent.name)
        return original_read_one(file, country, encoding)

    monkeypatch.setattr(article_module, "_read_one", recording_read_one)

    kwargs = (
        {"baseline": baseline}
        if reuse_baseline
        else {"baseline_start": "2020-01", "baseline_end": "2020-02"}
    )

    result, _ = ni.run_index_with_baseline(
        archive,
        DATA_DICTIONARY,
        windows=[("2020-03", "2020-03")],
        countries=["Nigeria"],
        verbose=False,
        **kwargs,
    )

    assert set(result["window"]) == {"2020-03:2020-03"}

    expected = (
        {"2020-03"}
        if reuse_baseline
        else {"2020-01", "2020-02", "2020-03"}
    )

    assert set(opened_months) == expected, (
        f"Opened {sorted(set(opened_months))}; "
        f"expected only {sorted(expected)}"
    )

@pytest.mark.parametrize("reuse_baseline", [False, True], ids=["fresh", "reused"])
def test_baseline_optimization_preserves_epu_values(tmp_path, monkeypatch, reuse_baseline):
    """Selective month reading must produce identical EPU results."""
    archive = tmp_path / "archive"
    _build_archive(archive)

    # Include irrelevant months and a gap between the required periods.
    for month in ("2015-06", "2022-06", "2024-06"):
        _write(
            archive / "Nigeria" / "The Punch" / month
            / f"articles_The Punch_{month}.csv.gz",
            [
                f'The Punch,{month}-01,"Government fiscal policy creates uncertainty",'
                f'https://example.com/nigeria/{month}/old,body'
            ],
        )

    import newsindex.pipeline as pipeline

    original_read_and_count = pipeline._read_and_count

    # Simulate the original implementation by ignoring date windows.
    def unfiltered_read_and_count(root, dictionary_path, countries, **kwargs):
        kwargs.pop("windows", None)
        return original_read_and_count(
            root, dictionary_path, countries, **kwargs
        )

    kwargs = {
        "windows": [("2020-03", "2020-03")],
        "countries": ["Nigeria"],
        "verbose": False,
    }

    if reuse_baseline:
        keywords = ni.dictionary.load(DATA_DICTIONARY)
        counts = ni.index.count(
            ni.dedup.dedupe(ni.read_articles(archive)),
            keywords,
        )
        baseline = ni.index.fit_baseline(
            counts, start="2020-01", end="2020-02"
        )
        kwargs["baseline"] = baseline
    else:
        kwargs["baseline_start"] = "2020-01"
        kwargs["baseline_end"] = "2020-02"

    # Run the optimized implementation.
    optimized_result, optimized_baseline = ni.run_index_with_baseline(
        archive, DATA_DICTIONARY, **kwargs
    )

    # Run with the original, unrestricted archive-reading behavior.
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "_read_and_count", unfiltered_read_and_count)
        original_result, original_baseline = ni.run_index_with_baseline(
            archive, DATA_DICTIONARY, **kwargs
        )

    # Compare the actual EPU results.
    pd.testing.assert_frame_equal(
        optimized_result,
        original_result,
    )

    # Compare baseline parameters when fitting a fresh baseline.
    if not reuse_baseline:
        assert optimized_baseline.start == original_baseline.start
        assert optimized_baseline.end == original_baseline.end


def test_run_index_with_baseline_skips_months_between_windows(
    tmp_path, monkeypatch
):
    """Only baseline and scoring months should be opened."""
    archive = tmp_path / "archive"
    _build_archive(archive)

    # Add an irrelevant month between the requested periods.
    for month in ("2020-06", "2022-01", "2024-06"):
        _write(
            archive / "Nigeria" / "The Punch" / month
            / f"articles_The Punch_{month}.csv.gz",
            [
                f'The Punch,{month}-01,"Government fiscal policy uncertainty",'
                f'https://example.com/nigeria/{month}/article,body'
            ],
        )

    import newsindex.articles as article_module

    opened_months = []
    original_read_one = article_module._read_one

    def recording_read_one(file, country, encoding):
        opened_months.append(file.parent.name)
        return original_read_one(file, country, encoding)

    monkeypatch.setattr(
        article_module, "_read_one", recording_read_one
    )

    result, _ = ni.run_index_with_baseline(
        archive,
        DATA_DICTIONARY,
        windows=[
            ("2020-03", "2020-03"),
            ("2024-06", "2024-06"),
        ],
        baseline_start="2020-01",
        baseline_end="2020-02",
        countries=["Nigeria"],
        verbose=False,
    )

    assert set(opened_months) == {
        "2020-01",
        "2020-02",
        "2020-03",
        "2024-06",
    }

    assert set(result["window"]) == {
        "2020-03:2020-03",
        "2024-06:2024-06",
    }

def test_run_index_with_baseline_empty_windows(tmp_path):
    """An empty scoring-window list should be handled consistently."""
    archive = tmp_path / "archive"
    _build_archive(archive)

    keywords = ni.dictionary.load(DATA_DICTIONARY)
    articles = ni.read_articles(archive)
    counts = ni.index.count(ni.dedup.dedupe(articles), keywords)

    baseline = ni.index.fit_baseline(
        counts, start="2020-01", end="2020-02"
    )

    result, returned_baseline = ni.run_index_with_baseline(
        archive,
        DATA_DICTIONARY,
        windows=[],
        baseline=baseline,
        countries=["Nigeria"],
        verbose=False,
    )

    assert result.empty
    assert returned_baseline is baseline