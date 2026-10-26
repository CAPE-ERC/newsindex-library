import gzip
from pathlib import Path

import pandas as pd
import pytest

from newsindex import read_articles
from newsindex.articles import iter_country_month_batches


def _write_articles(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "site,date,title,url,text\n" + "\n".join(rows)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(body)


def test_read_articles_walks_country_newspaper_month_layout_and_unescapes_html(tmp_path):
    _write_articles(
        tmp_path / "Nigeria" / "The Punch" / "2026-03" / "articles_The Punch_2026-03.csv.gz",
        ['The Punch,2026-03-01,"Government&#8217;s policy shift",https://example.com/a,"Full text"'],
    )
    _write_articles(
        tmp_path / "Kenya" / "Daily Nation" / "2026-03" / "articles_Daily Nation_2026-03.csv.gz",
        ['Daily Nation,2026-03-02,Economy grows,https://example.com/b,More text'],
    )

    articles = read_articles(tmp_path)

    assert len(articles) == 2
    assert set(articles["country"]) == {"Nigeria", "Kenya"}
    assert set(articles["year_month"]) == {"2026-03"}
    row = articles[articles["country"] == "Nigeria"].iloc[0]
    assert row["title"] == "Government’s policy shift"


def test_read_articles_returns_categorical_country_site_year_month(tmp_path):
    _write_articles(
        tmp_path / "Nigeria" / "The Punch" / "2026-03" / "articles_The Punch_2026-03.csv.gz",
        ['The Punch,2026-03-01,Headline,https://example.com/a,Text'],
    )

    articles = read_articles(tmp_path)

    for col in ("country", "site", "year_month"):
        assert isinstance(articles[col].dtype, pd.CategoricalDtype)


def test_read_articles_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_articles(tmp_path)


def test_read_articles_countries_filters_to_the_named_folders(tmp_path):
    _write_articles(
        tmp_path / "Nigeria" / "The Punch" / "2026-03" / "articles_The Punch_2026-03.csv.gz",
        ['The Punch,2026-03-01,Headline,https://example.com/a,Text'],
    )
    _write_articles(
        tmp_path / "Kenya" / "Daily Nation" / "2026-03" / "articles_Daily Nation_2026-03.csv.gz",
        ['Daily Nation,2026-03-02,Economy grows,https://example.com/b,More text'],
    )

    articles = read_articles(tmp_path, countries=["nigeria"])  # case-insensitive match

    assert set(articles["country"]) == {"Nigeria"}


def test_read_articles_countries_all_reads_every_folder(tmp_path):
    _write_articles(
        tmp_path / "Nigeria" / "The Punch" / "2026-03" / "articles_The Punch_2026-03.csv.gz",
        ['The Punch,2026-03-01,Headline,https://example.com/a,Text'],
    )
    _write_articles(
        tmp_path / "Kenya" / "Daily Nation" / "2026-03" / "articles_Daily Nation_2026-03.csv.gz",
        ['Daily Nation,2026-03-02,Economy grows,https://example.com/b,More text'],
    )

    articles = read_articles(tmp_path, countries="all")

    assert set(articles["country"]) == {"Nigeria", "Kenya"}


def test_read_articles_countries_raises_on_unknown_folder(tmp_path):
    _write_articles(
        tmp_path / "Nigeria" / "The Punch" / "2026-03" / "articles_The Punch_2026-03.csv.gz",
        ['The Punch,2026-03-01,Headline,https://example.com/a,Text'],
    )

    with pytest.raises(ValueError, match="Ghana"):
        read_articles(tmp_path, countries=["Nigeria", "Ghana"])


def _build_multi_month_archive(root: Path) -> None:
    _write_articles(
        root / "Nigeria" / "The Punch" / "2026-01" / "articles_The Punch_2026-01.csv.gz",
        ['The Punch,2026-01-05,Jan headline,https://example.com/a,Text A'],
    )
    _write_articles(
        root / "Nigeria" / "Vanguard" / "2026-01" / "articles_Vanguard_2026-01.csv.gz",
        ['Vanguard,2026-01-06,Jan headline 2,https://example.com/b,Text B'],
    )
    _write_articles(
        root / "Nigeria" / "The Punch" / "2026-02" / "articles_The Punch_2026-02.csv.gz",
        ['The Punch,2026-02-10,Feb headline,https://example.com/c,Text C'],
    )
    _write_articles(
        root / "Kenya" / "Daily Nation" / "2026-01" / "articles_Daily Nation_2026-01.csv.gz",
        ['Daily Nation,2026-01-07,Kenya headline,https://example.com/d,Text D'],
    )


def test_iter_country_month_batches_yields_one_batch_per_country_month(tmp_path):
    _build_multi_month_archive(tmp_path)

    batches = list(iter_country_month_batches(tmp_path))

    # Nigeria/2026-01 (2 newspapers combined), Nigeria/2026-02, Kenya/2026-01
    assert len(batches) == 3
    sizes = sorted(len(b) for b in batches)
    assert sizes == [1, 1, 2]
    for batch in batches:
        assert batch["country"].nunique() == 1


def test_iter_country_month_batches_matches_read_articles_when_concatenated(tmp_path):
    _build_multi_month_archive(tmp_path)

    streamed = pd.concat(list(iter_country_month_batches(tmp_path)), ignore_index=True)
    whole = read_articles(tmp_path)

    sort_cols = ["country", "site", "date", "url"]
    streamed_sorted = streamed.sort_values(sort_cols).reset_index(drop=True)
    whole_sorted = whole.sort_values(sort_cols).reset_index(drop=True)
    for col in streamed_sorted.columns:
        assert list(streamed_sorted[col].astype(str)) == list(whole_sorted[col].astype(str))


def test_iter_country_month_batches_countries_filters(tmp_path):
    _build_multi_month_archive(tmp_path)

    batches = list(iter_country_month_batches(tmp_path, countries=["nigeria"]))

    assert all((b["country"] == "Nigeria").all() for b in batches)
    assert sum(len(b) for b in batches) == 3


def test_iter_country_month_batches_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_country_month_batches(tmp_path))
