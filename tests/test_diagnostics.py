"""coverage / validate_archive / dictionary_coverage / health_check."""

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
    for month, epu_count, plain_count in [("2020-01", 1, 3), ("2020-02", 2, 2)]:
        rows = [
            f'The Punch,{month}-01,"Government fiscal policy and inflation create uncertainty",'
            f"https://example.com/nigeria/{month}/{i},body"
            for i in range(epu_count)
        ] + [
            f'The Punch,{month}-02,"Local football result {i}",https://example.com/nigeria/{month}/p{i},body'
            for i in range(plain_count)
        ]
        _write(root / "Nigeria" / "The Punch" / month / f"articles_The Punch_{month}.csv.gz", rows)

    for month, epu_count, plain_count in [("2020-01", 2, 2), ("2020-02", 1, 3)]:
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
def archive(tmp_path) -> Path:
    root = tmp_path / "archive"
    _build_archive(root)
    return root


def test_coverage_reports_one_row_per_country_month(archive):
    cov = ni.coverage(archive)

    assert set(cov["country"]) == {"Nigeria", "Kenya"}
    assert set(cov["year_month"]) == {"2020-01", "2020-02"}
    assert (cov["n_files"] == 1).all()
    assert (cov["n_articles"] > 0).all()


def test_coverage_respects_countries_filter(archive):
    cov = ni.coverage(archive, countries=["Nigeria"])

    assert set(cov["country"]) == {"Nigeria"}


def test_validate_archive_is_clean_on_a_well_formed_archive(archive):
    issues = ni.validate_archive(archive)

    assert issues.empty


def test_validate_archive_flags_a_missing_column(archive, tmp_path):
    bad_path = archive / "Nigeria" / "The Punch" / "2020-03"
    bad_path.mkdir(parents=True, exist_ok=True)
    (bad_path / "articles_The Punch_2020-03.csv").write_text(
        "site,date,title,url\nThe Punch,2020-03-01,Some title,https://example.com/x\n", encoding="utf-8"
    )

    issues = ni.validate_archive(archive, countries=["Nigeria"])

    assert (issues["issue"].str.contains("missing column")).any()


def test_validate_archive_flags_an_empty_file(archive):
    empty_path = archive / "Nigeria" / "The Punch" / "2020-04"
    empty_path.mkdir(parents=True, exist_ok=True)
    (empty_path / "articles_The Punch_2020-04.csv").write_text(
        "site,date,title,url,text\n", encoding="utf-8"
    )

    issues = ni.validate_archive(archive, countries=["Nigeria"])

    assert (issues["issue"] == "empty file (no data rows)").any()


def test_validate_archive_flags_a_date_folder_mismatch(archive):
    bad_path = archive / "Nigeria" / "The Punch" / "2020-05"
    bad_path.mkdir(parents=True, exist_ok=True)
    (bad_path / "articles_The Punch_2020-05.csv").write_text(
        "site,date,title,url,text\nThe Punch,2020-06-01,Some title,https://example.com/x,body\n",
        encoding="utf-8",
    )

    issues = ni.validate_archive(archive, countries=["Nigeria"])

    assert (issues["issue"].str.contains("date not matching folder")).any()


def test_dictionary_coverage_flags_a_country_missing_from_the_dictionary(archive):
    (archive / "Freedonia" / "The Gazette" / "2020-01").mkdir(parents=True)
    (archive / "Freedonia" / "The Gazette" / "2020-01" / "articles_The Gazette_2020-01.csv").write_text(
        "site,date,title,url,text\nThe Gazette,2020-01-01,Hello,https://example.com/x,body\n",
        encoding="utf-8",
    )
    kw = ni.dictionary.load(DATA_DICTIONARY)

    cov = ni.dictionary_coverage(kw, archive)

    freedonia = cov[cov["country"] == "Freedonia"].iloc[0]
    assert bool(freedonia["in_archive"]) is True
    assert bool(freedonia["in_dictionary"]) is False
    nigeria = cov[cov["country"] == "Nigeria"].iloc[0]
    assert bool(nigeria["in_archive"]) is True
    assert bool(nigeria["in_dictionary"]) is True


def test_health_check_reports_ok_true_on_a_clean_wired_up_archive(archive, capsys):
    summary = ni.health_check(archive, DATA_DICTIONARY, countries=["Nigeria", "Kenya"])

    assert summary["ok"] is True
    assert summary["countries_in_archive_not_dictionary"] == []
    assert summary["n_country_months"] == 4
    assert summary["n_issues"] == 0
    assert isinstance(summary["issues"], pd.DataFrame)
    assert capsys.readouterr().err


def test_health_check_reports_ok_false_when_a_country_is_missing_from_the_dictionary(archive):
    (archive / "Freedonia" / "The Gazette" / "2020-01").mkdir(parents=True)
    (archive / "Freedonia" / "The Gazette" / "2020-01" / "articles_The Gazette_2020-01.csv").write_text(
        "site,date,title,url,text\nThe Gazette,2020-01-01,Hello,https://example.com/x,body\n",
        encoding="utf-8",
    )

    summary = ni.health_check(archive, DATA_DICTIONARY, verbose=False)

    assert summary["ok"] is False
    assert "Freedonia" in summary["countries_in_archive_not_dictionary"]
