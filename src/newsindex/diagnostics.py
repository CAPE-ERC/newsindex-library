"""Catch problems in an archive or dictionary before spending a long run on bad input.

These operate on the archive/dictionary directly -- not on a result table --
so they're cheap to run before committing to a full
:func:`newsindex.run_index`/:func:`newsindex.run_index_with_baseline` call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import dictionary as _dictionary
from .articles import _REQUIRED_COLUMNS, _resolve_countries, iter_country_month_batches


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)


def coverage(root: str | Path, *, countries: list[str] | str | None = None) -> pd.DataFrame:
    """Which (country, year-month) combinations actually have article files under ``root``.

    One row per (country, year-month) that has at least one file, with
    ``n_files`` and ``n_articles`` (row count across those files). Reads
    every file's contents (via :func:`newsindex.articles.iter_country_month_batches`)
    to get an accurate article count, so this isn't free on a very large
    archive, but it never holds more than one country-month in memory at a
    time -- useful to run before a big call to catch gaps in advance.
    """
    rows: list[dict] = []
    for batch in iter_country_month_batches(root, countries=countries):
        if batch.empty:
            continue
        country = str(batch["country"].iloc[0])
        for year_month, month_rows in batch.groupby("year_month", observed=True):
            rows.append(
                {
                    "country": country,
                    "year_month": str(year_month),
                    "n_files": int(month_rows["site"].nunique()),
                    "n_articles": int(len(month_rows)),
                }
            )
    result = pd.DataFrame(rows, columns=["country", "year_month", "n_files", "n_articles"])
    return result.sort_values(["country", "year_month"]).reset_index(drop=True)


def _validate_file(file: Path, country: str) -> dict | None:
    """One issue row for ``file`` (``None`` if it's clean).

    Checks: failure to parse at all, missing required columns, zero data
    rows, and a ``date`` value that doesn't start with the year-month
    folder the file is stored in. Never raises -- a file that can't even
    be parsed becomes an ``issue`` row instead of an uncaught exception.
    """
    year_month_folder = file.parent.name
    try:
        frame = pd.read_csv(file, encoding="utf-8", compression="infer")
    except Exception as exc:  # noqa: BLE001 -- any parse failure is itself a reportable issue
        return {"country": country, "year_month": year_month_folder, "file": str(file), "issue": f"failed to parse: {exc}"}

    missing = [col for col in _REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        return {"country": country, "year_month": year_month_folder, "file": str(file), "issue": f"missing column(s): {missing}"}

    if frame.empty:
        return {"country": country, "year_month": year_month_folder, "file": str(file), "issue": "empty file (no data rows)"}

    mismatched = ~frame["date"].astype(str).str.startswith(year_month_folder)
    if mismatched.any():
        return {
            "country": country,
            "year_month": year_month_folder,
            "file": str(file),
            "issue": f"{int(mismatched.sum())} row(s) with date not matching folder {year_month_folder!r}",
        }
    return None


def _archive_files(root: Path, country: str) -> list[Path]:
    country_root = root / country
    return sorted({*country_root.glob("*/*/*.csv.gz"), *country_root.glob("*/*/*.csv")})


def validate_archive(root: str | Path, *, countries: list[str] | str | None = None) -> pd.DataFrame:
    """Sanity-check every file under ``root`` without any EPU classification.

    Checks, per file: missing required columns, zero data rows, and a
    ``date`` value that doesn't start with the year-month folder it's
    stored in. Never raises on a malformed file -- a file that can't even
    be parsed becomes one ``issue`` row instead of an uncaught exception.

    Returns one row per issue found (``country``, ``year_month``, ``file``,
    ``issue``) -- an empty DataFrame if the archive is clean.
    """
    root = Path(root)
    selected = _resolve_countries(root, countries)

    issues: list[dict] = []
    for country in selected:
        for file in _archive_files(root, country):
            issue = _validate_file(file, country)
            if issue is not None:
                issues.append(issue)

    result = pd.DataFrame(issues, columns=["country", "year_month", "file", "issue"])
    return result.sort_values(["country", "year_month", "file"]).reset_index(drop=True)


def dictionary_coverage(
    dictionary: _dictionary.Dictionary, root: str | Path, *, countries: list[str] | str | None = None
) -> pd.DataFrame:
    """Which countries have an archive folder, a dictionary entry, or both.

    With ``countries=None``/``"all"`` (the default): one row per country in
    the union of every archive folder under ``root`` and every one of
    ``dictionary``'s countries. A country with ``in_archive=True,
    in_dictionary=False`` would fail at :func:`newsindex.index.count` with
    a ``KeyError`` -- catch it here first.

    With an explicit ``countries`` list: the comparison is scoped to just
    those countries (every one of which is guaranteed ``in_archive=True``,
    since :func:`newsindex.articles.read_articles`'s country-resolution
    raises on a name with no matching folder) -- a country outside the list
    isn't reported as "missing", since the caller deliberately scoped down.
    """
    root = Path(root)
    archive_countries = set(_resolve_countries(root, countries))
    dictionary_countries = set(dictionary.countries)

    if countries is None or countries == "all":
        all_countries = sorted(archive_countries | dictionary_countries)
    else:
        all_countries = sorted(archive_countries)

    result = pd.DataFrame(
        {
            "country": all_countries,
            "in_archive": [c in archive_countries for c in all_countries],
            "in_dictionary": [c in dictionary_countries for c in all_countries],
        }
    )
    return result


def health_check(
    root: str | Path,
    dictionary_path: str | Path,
    *,
    countries: list[str] | str | None = None,
    sample_files_per_country: int = 50,
    verbose: bool = True,
) -> dict:
    """One-call "is everything wired up correctly" smoke test before a big run.

    Loads the dictionary, compares it against the archive
    (:func:`dictionary_coverage`), measures what's actually present
    (:func:`coverage`), and validates a bounded sample of files per country
    (:func:`validate_archive`, capped at ``sample_files_per_country`` files
    per country so this stays cheap even on a huge archive -- pass a larger
    value, or run :func:`validate_archive` directly, for an exhaustive scan).

    Returns a dict: ``ok`` (bool, false if any dictionary mismatch or
    validation issue was found), ``countries_in_archive_not_dictionary``,
    ``countries_in_dictionary_not_archive``, ``n_country_months``,
    ``n_issues``, ``issues`` (the sampled :func:`validate_archive` table).
    """
    path = Path(dictionary_path)
    if not path.exists():
        raise FileNotFoundError(f"dictionary file not found: {path}")
    _log(verbose, f"Loading dictionary from {path} ...")
    kw = _dictionary.load(path)

    _log(verbose, "Checking dictionary coverage against the archive ...")
    dict_cov = dictionary_coverage(kw, root, countries=countries)
    missing_from_dictionary = sorted(
        dict_cov.loc[dict_cov["in_archive"] & ~dict_cov["in_dictionary"], "country"]
    )
    missing_from_archive = sorted(
        dict_cov.loc[dict_cov["in_dictionary"] & ~dict_cov["in_archive"], "country"]
    )

    _log(verbose, "Measuring archive coverage ...")
    cov = coverage(root, countries=countries)

    _log(verbose, f"Validating up to {sample_files_per_country} file(s) per country ...")
    root_path = Path(root)
    selected = _resolve_countries(root_path, countries)
    sampled_issues: list[dict] = []
    for country in selected:
        for file in _archive_files(root_path, country)[:sample_files_per_country]:
            issue = _validate_file(file, country)
            if issue is not None:
                sampled_issues.append(issue)

    issues = pd.DataFrame(sampled_issues, columns=["country", "year_month", "file", "issue"])

    ok = not missing_from_dictionary and not missing_from_archive and issues.empty
    summary = {
        "ok": ok,
        "countries_in_archive_not_dictionary": missing_from_dictionary,
        "countries_in_dictionary_not_archive": missing_from_archive,
        "n_country_months": int(len(cov)),
        "n_issues": int(len(issues)),
        "issues": issues,
    }

    if verbose:
        _log(verbose, f"health_check: ok={ok}, {summary['n_country_months']} country-month(s) found, {summary['n_issues']} issue(s) in sample.")
        if missing_from_dictionary:
            _log(verbose, f"  archive countries with no dictionary entry: {missing_from_dictionary}")
        if missing_from_archive:
            _log(verbose, f"  dictionary countries with no archive folder: {missing_from_archive}")

    return summary
