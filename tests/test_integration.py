"""End-to-end smoke test following the README usage example, with a tiny
synthetic archive and the real production keyword dictionary."""

import gzip
from pathlib import Path

import newsindex as ni

DATA_DICTIONARY = Path(__file__).resolve().parents[1] / "data" / "eptu_country_keywords.json"
DATA_GDP = Path(__file__).resolve().parents[1] / "data" / "gdp_ppp_2015_2025.csv"


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


def test_full_pipeline_runs_end_to_end(tmp_path):
    _build_archive(tmp_path / "archive")

    kw = ni.dictionary.load(DATA_DICTIONARY)
    arts = ni.read_articles(tmp_path / "archive")
    deduped = ni.dedup.dedupe(arts)
    counts = ni.index.count(deduped, kw)

    baseline = ni.index.fit_baseline(counts, start="2020-01", end="2020-02")
    baseline_path = tmp_path / "baseline.json"
    baseline.save(baseline_path)
    reloaded = ni.index.Baseline.load(baseline_path)

    country = ni.index.score(counts, reloaded)
    assert set(country["country"]) <= {"Nigeria", "Kenya"}
    assert (country["EPU"] >= 0).all()

    gdp = ni.aggregate.monthly_gdp(DATA_GDP)
    groups = {"WestAfrica": ["Nigeria"], "EastAfrica": ["Kenya"]}
    regions = ni.aggregate.aggregate(
        country, gdp, groups=groups, baseline=reloaded, include_africa=True
    )
    assert {"WestAfrica", "EastAfrica", "Africa"} <= set(regions["group"])
    assert regions["index"].notna().any()
