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

from ._version import __version__
from .dictionary import Dictionary
from .match import is_epu


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
    """
    missing = sorted(set(articles[country_col]) - set(dictionary.countries))
    if missing:
        raise KeyError(f"countries missing from dictionary: {missing}")

    def _is_epu(row: pd.Series) -> bool:
        entry = dictionary[row[country_col]]
        return is_epu(row[title_col], row[text_col], entry.economic, entry.policy, entry.uncertainty)

    working = articles.copy()
    working["_is_epu"] = working.apply(_is_epu, axis=1)
    grouped = (
        working.groupby([country_col, site_col, month_col])["_is_epu"]
        .agg(epu_articles="sum", all_articles="count")
        .reset_index()
    )
    grouped = grouped.rename(
        columns={country_col: "country", site_col: "site", month_col: "year_month"}
    )
    return grouped.sort_values(["country", "site", "year_month"]).reset_index(drop=True)


@dataclass(frozen=True)
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
