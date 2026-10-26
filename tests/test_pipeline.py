"""run_index / run_index_with_baseline: one-call orchestration."""

import gzip
from pathlib import Path

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
