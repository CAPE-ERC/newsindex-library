"""Article deduplication within a country-month.

Articles are deduplicated primarily by canonical URL. When the URL is
missing or empty, a hash of the normalized title and text stands in for it,
so two copies of the same story with no URL still collapse to one row.
"""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

from .match import is_missing, normalize

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}


def canonical_url(url: object) -> str:
    """Normalize a URL for deduplication: lower-cased scheme/host, no
    fragment, no tracking query parameters, no trailing slash."""
    text = "" if is_missing(url) else str(url).strip()
    if not text:
        return ""
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or ""
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(query_pairs))
    return urlunsplit((scheme, netloc, path, query, ""))


def _hash_blob(blob: str) -> str:
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dedupe(
    articles: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("country", "year_month"),
    url_col: str = "url",
    title_col: str = "title",
    text_col: str = "text",
) -> pd.DataFrame:
    """Drop duplicate articles within each ``group_cols`` group.

    A row is a duplicate of an earlier row in the same group if they share a
    canonical URL, or, when either URL is missing, if they share a hash of
    their normalized title and text. The first occurrence in the input is
    kept.
    """
    if articles.empty:
        return articles.copy()

    canonical = articles[url_col].map(canonical_url)
    blob = articles[title_col].map(normalize) + "\x1f" + articles[text_col].map(normalize)
    fallback = blob.map(_hash_blob)
    dedup_key = canonical.where(canonical != "", fallback)

    working = articles.copy()
    working["_dedup_key"] = dedup_key
    missing_group_cols = [col for col in group_cols if col not in working.columns]
    if missing_group_cols:
        raise KeyError(f"missing grouping column(s): {missing_group_cols}")

    deduped = working.drop_duplicates(subset=[*group_cols, "_dedup_key"], keep="first")
    return deduped.drop(columns="_dedup_key").reset_index(drop=True)
