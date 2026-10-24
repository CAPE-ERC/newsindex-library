"""Read scraped article archives into a single table.

Expects the on-disk layout produced by CAPE's scraper: ``<root>/<country>/
<newspaper>/<year-month>/articles_<newspaper>_<year-month>.csv(.gz)``, each
file holding the columns ``site``, ``date``, ``title``, ``url``, ``text``.
This is the one function in the library that takes a path rather than a
table, matching the documented example usage.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = ("site", "date", "title", "url", "text")


def _unescape(value: object) -> object:
    if not isinstance(value, str):
        return value
    return html.unescape(value)


def read_articles(root: str | Path, *, encoding: str = "utf-8") -> pd.DataFrame:
    """Read every ``articles_*.csv`` / ``articles_*.csv.gz`` file under
    ``root`` into one table, adding ``country`` (the top-level folder name)
    and ``year_month`` (the first seven characters of ``date``) columns.

    Article titles and text are HTML-unescaped (the scraper stores raw HTML
    entities such as ``&#8217;``) before they reach :mod:`newsindex.match`.
    """
    root = Path(root)
    files = sorted({*root.glob("**/*.csv.gz"), *root.glob("**/*.csv")})
    if not files:
        raise FileNotFoundError(f"no article files found under {root}")

    frames: list[pd.DataFrame] = []
    for file in files:
        country = file.relative_to(root).parts[0]
        frame = pd.read_csv(file, encoding=encoding, compression="infer")
        missing = [col for col in _REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"{file}: missing column(s) {missing}")
        frame["country"] = country
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["title"] = combined["title"].map(_unescape)
    combined["text"] = combined["text"].map(_unescape)
    combined["year_month"] = combined["date"].astype(str).str.slice(0, 7)
    return combined[["country", "site", "year_month", "date", "title", "url", "text"]]
