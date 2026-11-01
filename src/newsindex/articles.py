"""Read scraped article archives into tables.

Expects the on-disk layout produced by CAPE's scraper: ``<root>/<country>/
<newspaper>/<year-month>/articles_<newspaper>_<year-month>.csv(.gz)``, each
file holding the columns ``site``, ``date``, ``title``, ``url``, ``text``.
These are the only functions in the library that take a path rather than a
table, matching the documented example usage.

:func:`read_articles` materializes the whole requested scope as one table,
which is simple but memory-heavy for a large archive (full article text for
every country/newspaper/month at once). :func:`iter_country_month_batches`
is the memory-safe alternative, yielding one small batch at a time; it's
what :func:`newsindex.index.count_archive` (and therefore
:func:`newsindex.run_index`/:func:`newsindex.run_index_with_baseline`) use
for real production-scale runs.
"""

from __future__ import annotations

import html
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = ("site", "date", "title", "url", "text")


def _unescape(value: object) -> object:
    if not isinstance(value, str):
        return value
    return html.unescape(value)


def _resolve_countries(root: Path, countries: list[str] | str | None) -> list[str]:
    available = sorted(p.name for p in root.iterdir() if p.is_dir())
    if countries is None or countries == "all":
        return available

    by_lower = {name.lower(): name for name in available}
    selected, missing = [], []
    for name in countries:
        match = by_lower.get(name.lower())
        (selected if match else missing).append(match or name)
    if missing:
        raise ValueError(
            f"country folder(s) not found under {root}: {missing}. Available: {available}"
        )
    return selected


def _read_one(file: Path, country: str, encoding: str) -> pd.DataFrame:
    frame = pd.read_csv(file, encoding=encoding, compression="infer")
    missing = [col for col in _REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{file}: missing column(s) {missing}")
    frame["country"] = country
    return frame


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["title"] = frame["title"].map(_unescape)
    frame["text"] = frame["text"].map(_unescape)
    frame["year_month"] = frame["date"].astype(str).str.slice(0, 7)
    frame = frame[["country", "site", "year_month", "date", "title", "url", "text"]]
    for col in ("country", "site", "year_month"):
        frame[col] = frame[col].astype("category")
    return frame


def read_articles(
    root: str | Path, *, countries: list[str] | str | None = None, encoding: str = "utf-8"
) -> pd.DataFrame:
    """Read every ``articles_*.csv`` / ``articles_*.csv.gz`` file under
    ``root`` into one table, adding ``country`` (the top-level folder name)
    and ``year_month`` (the first seven characters of ``date``) columns.

    ``countries`` restricts which top-level country folders are read:
    ``None`` or ``"all"`` (the default) reads every folder under ``root``;
    a list reads only the named folders (matched case-insensitively
    against the folder names actually present, since archive folder names
    don't always match a dictionary's or :func:`newsindex.countries`'
    casing). An unmatched name raises :class:`ValueError` listing the
    folders that were found, to catch typos early.

    Article titles and text are HTML-unescaped (the scraper stores raw HTML
    entities such as ``&#8217;``) before they reach :mod:`newsindex.match`.
    ``country``/``site``/``year_month`` are returned as categorical columns
    (each distinct value stored once, not once per row) to keep the
    combined table's memory footprint down.

    This reads the whole requested scope into memory at once. For a large
    archive, prefer :func:`iter_country_month_batches` (or
    :func:`newsindex.index.count_archive`, which uses it), which never
    holds more than one country-month of articles at a time.
    """
    root = Path(root)
    selected = _resolve_countries(root, countries)

    files: list[Path] = []
    for country in selected:
        country_root = root / country
        files.extend(country_root.glob("**/*.csv.gz"))
        files.extend(country_root.glob("**/*.csv"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no article files found under {root}")

    frames = [_read_one(file, file.relative_to(root).parts[0], encoding) for file in files]
    return _finalize(pd.concat(frames, ignore_index=True))


def iter_country_month_batches(
    root: str | Path,
    *,
    countries: list[str] | str | None = None,
    encoding: str = "utf-8",
    start: str | None = None,
    end: str | None = None,
    windows: list[tuple[str, str]] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield one small table per (country, year-month folder) under ``root``,
    instead of materializing the whole requested scope in memory at once.

    Batching follows the archive's own layout: every newspaper's files
    under one ``<country>/*/<year-month>/`` folder are read and yielded
    together, then dropped before moving to the next (country, year-month)
    pair -- so peak memory stays close to the size of one country-month's
    articles, regardless of how many countries, newspapers, or years the
    overall scope spans. Unlike :func:`read_articles`, this requires the
    exact documented two-level ``<newspaper>/<year-month>/`` nesting under
    each country folder, since the year-month *folder* name is what
    batches are grouped by; each row's own ``year_month`` column is still
    derived from its actual ``date`` value, exactly as in
    :func:`read_articles` -- the folder name only decides batch boundaries.

    ``countries`` is as in :func:`read_articles`. ``start``/``end``, if
    given, restrict which year-month *folders* are read at all (inclusive
    ``"YYYY-MM"`` strings, compared lexicographically against the folder
    name) -- a month folder outside that range is skipped before any file
    in it is opened, instead of being read and discarded later. Raises
    :class:`FileNotFoundError` if nothing is found under ``root`` -- on
    first iteration, since this is a generator and nothing runs until then.
    """
    root = Path(root)
    selected = _resolve_countries(root, countries)

    found_any = False
    for country in selected:
        country_root = root / country
        files_by_month: dict[str, list[Path]] = {}
        for file in {*country_root.glob("*/*/*.csv.gz"), *country_root.glob("*/*/*.csv")}:
            year_month = file.parent.name
            if start is not None and year_month < start:
                continue
            if end is not None and year_month > end:
                continue
            if windows is not None and not any(
                window_start <= year_month <= window_end
                for window_start, window_end in windows
            ):
                continue
            files_by_month.setdefault(year_month, []).append(file)

        for year_month in sorted(files_by_month):
            files = sorted(files_by_month[year_month])
            frames = [_read_one(file, country, encoding) for file in files]
            found_any = True
            yield _finalize(pd.concat(frames, ignore_index=True))

    if not found_any:
        raise FileNotFoundError(f"no article files found under {root}")
