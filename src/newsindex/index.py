"""Count EPU articles per newspaper-month, fit the frozen baseline, and score
new months.

This module implements the Baker, Bloom and Davis (2016) scaling method
documented in the CAPE Africa EPU methodology: for newspaper *i* in month
*t*, ``S_it`` is the EPU article share, ``N_it = S_it / sigma_i`` standardises
it against the newspaper's baseline standard deviation, ``M_t`` is the mean
of ``N_it`` across newspapers, and ``EPU_t = 100 * M_t / mu`` rescales to a
baseline mean of 100. ``sigma_i``, ``mu``, and the retained newspapers are
frozen once at :func:`fit_baseline` time; :func:`score` never recomputes
them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from collections.abc import Callable

from . import dedup as _dedup
from ._version import __version__
from .articles import iter_country_month_batches
from .dictionary import Dictionary
from .match import component_pattern, loose_component_pattern, normalize


def _component_survivors(text: pd.Series, terms: tuple[str, ...]) -> pd.Index:
    """Rows of ``text`` whose article actually contains one of ``terms``.

    Two-stage: :func:`newsindex.match.loose_component_pattern` first (cheap,
    over-inclusive -- no word-boundary checks) to narrow down to candidates,
    then :func:`newsindex.match.component_pattern` (the precise, boundary-
    checked version) only on that narrowed subset. Equivalent to running the
    precise pattern over ``text`` directly, just far cheaper when most rows
    don't contain any of ``terms`` at all, since the expensive pattern then
    only ever runs on the much smaller candidate subset.
    """
    if text.empty:
        return text.index[:0]
    candidates = text.index[text.str.contains(loose_component_pattern(terms), regex=True, na=False)]
    if candidates.empty:
        return candidates
    precise_hits = text.loc[candidates].str.contains(component_pattern(terms), regex=True, na=False)
    return candidates[precise_hits.to_numpy()]


def count(
    articles: pd.DataFrame,
    dictionary: Dictionary,
    *,
    country_col: str = "country",
    site_col: str = "site",
    month_col: str = "year_month",
    title_col: str = "title",
    text_col: str = "text",
) -> pd.DataFrame:
    """Count EPU and total articles per country-site-month.

    Every article's country must be present in ``dictionary``; a
    :class:`KeyError` is raised naming any country that is not, since a
    missing country silently produces an undercount rather than a visible
    failure.

    Classification is vectorized and short-circuits: per country, each
    component (Economic, Policy, Uncertainty) is checked only against the
    rows that survived the previous component, since EPU requires all
    three. The classification itself uses a cheap prefilter before the
    precise, word-boundary-checked pattern -- see
    :func:`newsindex.match.loose_component_pattern`. Together this avoids
    both the per-row/per-term Python loop and the full boundary-checked
    regex cost on articles that plainly don't qualify, which is most of
    them.
    """
    missing = sorted(set(articles[country_col]) - set(dictionary.countries))
    if missing:
        raise KeyError(f"countries missing from dictionary: {missing}")

    working = articles.copy()
    raw_text = working[title_col].fillna("").astype(str) + " " + working[text_col].fillna("").astype(str)
    normalized = raw_text.map(normalize)

    is_epu_flags = pd.Series(False, index=working.index)
    for country, rows in working.groupby(country_col, sort=False, observed=True).groups.items():
        entry = dictionary[country]
        survivors = rows
        for terms in (entry.economic, entry.policy, entry.uncertainty):
            if len(survivors) == 0:
                break
            survivors = _component_survivors(normalized.loc[survivors], terms)
        is_epu_flags.loc[survivors] = True

    working["_is_epu"] = is_epu_flags
    grouped = (
        working.groupby([country_col, site_col, month_col], sort=False, observed=True)["_is_epu"]
        .agg(epu_articles="sum", all_articles="count")
        .reset_index()
    )
    grouped = grouped.rename(
        columns={country_col: "country", site_col: "site", month_col: "year_month"}
    )
    # Categorical dtypes (e.g. from read_articles()) are a memory optimization
    # for the large per-article `working` table above; counts are already a
    # tiny aggregate, and downstream code (fit_baseline, score, aggregate,
    # ...) compares year_month with < / >=, which an unordered Categorical
    # doesn't support -- so plain strings from here on.
    for col in ("country", "site", "year_month"):
        grouped[col] = grouped[col].astype(str)
    return grouped.sort_values(["country", "site", "year_month"]).reset_index(drop=True)


def count_archive(
    root: str | Path,
    dictionary: Dictionary,
    *,
    countries: list[str] | str | None = None,
    encoding: str = "utf-8",
    dedupe: bool = True,
    country_col: str = "country",
    site_col: str = "site",
    month_col: str = "year_month",
    title_col: str = "title",
    text_col: str = "text",
    progress: Callable[[str], None] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Count EPU and total articles per country-site-month directly from an
    article archive, without ever materializing the whole requested scope
    in memory at once.

    Equivalent to ``count(dedupe(read_articles(root, countries=countries)),
    dictionary)`` with the default ``dedupe=True``, but processes one
    (country, year-month) batch at a time -- see
    :func:`newsindex.articles.iter_country_month_batches` -- discarding
    each batch's raw article text as soon as it's been counted. Peak memory
    stays close to the size of one country-month's articles, regardless of
    how many countries, newspapers, or years ``root`` spans. This is what
    :func:`newsindex.run_index` and :func:`newsindex.run_index_with_baseline`
    use internally.

    Deduplication happens within each (country, year-month folder) batch,
    matching the archive's own month-folder organization -- equivalent to
    deduplicating the whole requested scope at once, unless an article's
    actual date disagrees with the month folder it's stored in, in which
    case only within-batch duplicates are caught. Pass ``dedupe=False`` to
    skip this and deduplicate the full table yourself instead, if that
    distinction matters for your archive.

    ``progress``, if given, is called with one short status line after each
    batch (e.g. to print it) -- useful for a long run against a large
    archive, so it's visible as working rather than looking stalled.

    ``start``/``end``, if given, restrict which year-month folders are read
    at all (see :func:`newsindex.articles.iter_country_month_batches`) --
    pass the window you actually need (e.g. the same ``start``/``end`` as
    :func:`newsindex.run_index`) so progress and the resulting counts never
    cover months outside it.
    """
    parts: list[pd.DataFrame] = []
    for batch in iter_country_month_batches(
        root, countries=countries, encoding=encoding, start=start, end=end
    ):
        country = str(batch[country_col].iloc[0]) if len(batch) else "?"
        months = sorted(str(m) for m in batch[month_col].unique()) if len(batch) else []
        n_read = len(batch)

        if dedupe:
            batch = _dedup.dedupe(batch, url_col="url", title_col=title_col, text_col=text_col)
        n_deduped = len(batch)

        batch_counts = count(
            batch,
            dictionary,
            country_col=country_col,
            site_col=site_col,
            month_col=month_col,
            title_col=title_col,
            text_col=text_col,
        )
        parts.append(batch_counts)

        if progress is not None:
            epu_total = int(batch_counts["epu_articles"].sum())
            month_label = ",".join(months) if months else "?"
            progress(
                f"{country} {month_label}: {n_read:,} articles "
                f"({n_deduped:,} after dedup), {epu_total:,} EPU"
            )

    combined = pd.concat(parts, ignore_index=True)
    result = combined.groupby(["country", "site", "year_month"], as_index=False)[
        ["epu_articles", "all_articles"]
    ].sum()
    return result.sort_values(["country", "site", "year_month"]).reset_index(drop=True)


@dataclass(frozen=True, repr=False)
class Baseline:
    """Frozen baseline scaling parameters for the country EPU index."""

    start: str
    end: str
    mu: dict[str, float]
    sigma: dict[str, dict[str, float]]
    dictionary_fingerprint: str = ""
    library_version: str = field(default_factory=lambda: __version__)

    def retained_sites(self, country: str) -> list[str]:
        return sorted(self.sigma.get(country, {}))

    def __repr__(self) -> str:
        countries = sorted(self.mu)
        if not countries:
            return f"Baseline({self.start}..{self.end}, 0 countries)"
        shown = countries[:5]
        details = ", ".join(f"{c} ({len(self.retained_sites(c))} sites)" for c in shown)
        if len(countries) > len(shown):
            details += f", ... +{len(countries) - len(shown)} more"
        return f"Baseline({self.start}..{self.end}, {len(countries)} countries, e.g. {details})"

    def save(self, path: str | Path) -> None:
        payload = {
            "start": self.start,
            "end": self.end,
            "dictionary_fingerprint": self.dictionary_fingerprint,
            "library_version": self.library_version,
            "countries": {
                country: {"mu": self.mu[country], "sites": self.sigma[country]}
                for country in sorted(self.mu)
            },
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        countries = payload["countries"]
        return cls(
            start=payload["start"],
            end=payload["end"],
            mu={country: row["mu"] for country, row in countries.items()},
            sigma={country: dict(row["sites"]) for country, row in countries.items()},
            dictionary_fingerprint=payload.get("dictionary_fingerprint", ""),
            library_version=payload.get("library_version", ""),
        )


def fit_baseline(
    counts: pd.DataFrame,
    start: str,
    end: str,
    *,
    dictionary_fingerprint: str = "",
) -> Baseline:
    """Fit newspaper standard deviations and country means over ``[start, end]``.

    A newspaper is retained for a country's baseline if it has at least two
    monthly observations in the window and a nonzero population standard
    deviation of its EPU article share; a newspaper with constant or
    single-month coverage cannot be standardised and is dropped.
    """
    window = counts[(counts["year_month"] >= start) & (counts["year_month"] <= end)].copy()
    window = window[window["all_articles"] > 0]
    window["S_it"] = window["epu_articles"] / window["all_articles"]

    sigma: dict[str, dict[str, float]] = {}
    for (country, site), rows in window.groupby(["country", "site"]):
        values = rows["S_it"].to_numpy()
        if values.size < 2:
            continue
        sd = float(np.std(values, ddof=0))
        if sd > 0:
            sigma.setdefault(country, {})[site] = sd

    mu: dict[str, float] = {}
    for country, site_sigma in sigma.items():
        country_rows = window[window["country"] == country]
        n_it = country_rows.assign(
            sigma=country_rows["site"].map(site_sigma)
        ).dropna(subset=["sigma"])
        n_it = n_it[n_it["site"].isin(site_sigma)]
        n_it["N_it"] = n_it["S_it"] / n_it["site"].map(site_sigma)
        m_t = n_it.groupby("year_month")["N_it"].mean()
        if len(m_t):
            mu[country] = float(m_t.mean())

    countries_with_mu = set(mu)
    sigma = {country: sites for country, sites in sigma.items() if country in countries_with_mu}

    return Baseline(
        start=start,
        end=end,
        mu=mu,
        sigma=sigma,
        dictionary_fingerprint=dictionary_fingerprint,
    )


def score(counts: pd.DataFrame, baseline: Baseline) -> pd.DataFrame:
    """Score every country-month in ``counts`` against a frozen baseline."""
    working = counts[counts["all_articles"] > 0].copy()
    working["S_it"] = working["epu_articles"] / working["all_articles"]

    rows: list[dict] = []
    for country, mu in baseline.mu.items():
        site_sigma = baseline.sigma.get(country, {})
        country_rows = working[
            (working["country"] == country) & (working["site"].isin(site_sigma))
        ].copy()
        if country_rows.empty:
            continue
        country_rows["N_it"] = country_rows["S_it"] / country_rows["site"].map(site_sigma)
        for year_month, month_rows in country_rows.groupby("year_month"):
            m_t = float(month_rows["N_it"].mean())
            rows.append(
                {
                    "country": country,
                    "year_month": year_month,
                    "EPU": 100.0 * m_t / mu,
                    "M_t": m_t,
                    "mu": mu,
                    "n_sites": int(month_rows["site"].nunique()),
                }
            )
    result = pd.DataFrame(rows, columns=["country", "year_month", "EPU", "M_t", "mu", "n_sites"])
    return result.sort_values(["country", "year_month"]).reset_index(drop=True)


def compute_index(counts: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Compute an in-sample index over a single window (the no-baseline variant).

    Fits ``sigma``/``mu`` from ``[start, end]`` and scores that same window
    against them; nothing is frozen or reused for a later period. Equivalent
    to ``score(window, fit_baseline(window, start, end))`` where ``window``
    is ``counts`` restricted to ``[start, end]``, but callers never see or
    manage a :class:`Baseline`.
    """
    window = counts[(counts["year_month"] >= start) & (counts["year_month"] <= end)]
    baseline = fit_baseline(window, start, end)
    return score(window, baseline)


def score_windows(
    counts: pd.DataFrame, baseline: Baseline, windows: list[tuple[str, str]]
) -> pd.DataFrame:
    """Score one or more later windows against a frozen baseline (the
    with-baseline variant).

    ``windows`` is a list of inclusive ``(start, end)`` ``"YYYY-MM"`` pairs,
    typically just one period that follows the baseline window, but any
    number may be given; they may also overlap or leave gaps between them.
    The result adds a ``window`` column (``"start:end"``) identifying which
    pair produced each row.
    """
    parts: list[pd.DataFrame] = []
    for start, end in windows:
        window_counts = counts[(counts["year_month"] >= start) & (counts["year_month"] <= end)]
        scored = score(window_counts, baseline)
        scored = scored.assign(window=f"{start}:{end}")
        parts.append(scored)
    columns = ["window", "country", "year_month", "EPU", "M_t", "mu", "n_sites"]
    if not parts:
        return pd.DataFrame(columns=columns)
    result = pd.concat(parts, ignore_index=True)
    return result[columns].sort_values(["window", "country", "year_month"]).reset_index(drop=True)
