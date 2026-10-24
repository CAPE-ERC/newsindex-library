"""Country z-scores, GDP weights, and regional / Africa aggregation.

Implements the four computation steps from the CAPE Africa EPU methodology:
standardise each country's index against its own baseline, weight by
within-group GDP-PPP share, take the GDP-weighted average of country
z-scores, and rescale the result to a baseline mean of 100 and standard
deviation of 10. The Africa aggregate applies the same four steps across all
countries directly; it is not an average of the regional series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: The five CAPE African regions, as published in the Africa EPU Index
#: methodology (Table 8). Country names match the keys used in the
#: production ``eptu_country_keywords.json`` dictionary.
AFRICA_REGIONS: dict[str, list[str]] = {
    "Central": [
        "Cameroon",
        "Central African Republic",
        "Chad",
        "Congo",
        "Congo DRC",
        "Equatorial Guinea",
        "Gabon",
    ],
    "East": [
        "Burundi",
        "Comoros",
        "Djibouti",
        "Eritrea",
        "Ethiopia",
        "Kenya",
        "Rwanda",
        "Seychelles",
        "Somalia",
        "South Sudan",
        "Sudan",
        "Tanzania",
        "Uganda",
    ],
    "North": ["Algeria", "Egypt", "Libya", "Mauritania", "Morocco", "Tunisia"],
    "Southern": [
        "Angola",
        "Botswana",
        "Eswatini",
        "Lesotho",
        "Madagascar",
        "Malawi",
        "Mauritius",
        "Mozambique",
        "Namibia",
        "Sao Tome and Principe",
        "South Africa",
        "Zambia",
        "Zimbabwe",
    ],
    "West": [
        "Benin",
        "Burkina-Faso",
        "Cape Verde",
        "Gambia",
        "Ghana",
        "Guinea",
        "Guinea-Bissau",
        "Ivory Coast",
        "Liberia",
        "Mali",
        "Niger",
        "Nigeria",
        "Senegal",
        "Sierra Leone",
        "Togo",
    ],
}

#: Country-name aliases between common GDP source spellings and the names
#: used by :data:`AFRICA_REGIONS` and the EPU keyword dictionary.
DEFAULT_COUNTRY_ALIASES: dict[str, str] = {
    "Cabo Verde": "Cape Verde",
    "Burkina Faso": "Burkina-Faso",
    "Congo, The Democratic Republic of the": "Congo DRC",
    "Democratic Republic of the Congo": "Congo DRC",
    "Congo, Dem. Rep.": "Congo DRC",
    "Congo, Rep.": "Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Eswatini (Swaziland)": "Eswatini",
    "Sao Tome and Principe": "Sao Tome and Principe",
    "The Gambia": "Gambia",
}


def monthly_gdp(
    path: str | Path,
    *,
    country_col: str = "Country",
    year_col: str = "Year",
    value_col: str = "GDP_Billions_USD",
    scale: float = 1.0,
    end: str | None = None,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Interpolate annual GDP-PPP to monthly values.

    Each annual value is placed at 1 January of its year and interpolated
    linearly in logarithms through the next year's figure. After a
    country's last annual figure, the value is held flat (carried forward),
    optionally through ``end`` (a ``"YYYY-MM"`` string).

    ``scale`` multiplies every value after it is read, e.g. ``scale=1000``
    to convert a billions-denominated source file to millions.
    """
    aliases = DEFAULT_COUNTRY_ALIASES if aliases is None else aliases
    raw = pd.read_csv(path)
    raw[country_col] = raw[country_col].map(lambda c: aliases.get(c, c))

    rows: list[dict] = []
    for country, group in raw.groupby(country_col):
        pairs = sorted(
            (int(year), float(value) * scale)
            for year, value in zip(group[year_col], group[value_col])
        )
        for i, (year, value) in enumerate(pairs):
            if i + 1 < len(pairs):
                next_year, next_value = pairs[i + 1]
                if next_year == year:
                    continue
                for m in range(12):
                    frac = m / 12.0
                    if value > 0 and next_value > 0:
                        interpolated = float(
                            np.exp(np.log(value) + frac * (np.log(next_value) - np.log(value)))
                        )
                    else:
                        interpolated = value
                    rows.append(
                        {"country": country, "year_month": f"{year}-{m + 1:02d}", "gdp": interpolated}
                    )
            else:
                for m in range(12):
                    rows.append(
                        {"country": country, "year_month": f"{year}-{m + 1:02d}", "gdp": value}
                    )
                if end is not None:
                    carry_year = year + 1
                    while f"{carry_year}-01" <= end:
                        for m in range(12):
                            year_month = f"{carry_year}-{m + 1:02d}"
                            if year_month > end:
                                break
                            rows.append({"country": country, "year_month": year_month, "gdp": value})
                        carry_year += 1

    result = pd.DataFrame(rows, columns=["country", "year_month", "gdp"])
    return result.sort_values(["country", "year_month"]).reset_index(drop=True)


def _population_stats(series: pd.Series) -> tuple[float, float]:
    values = series.to_numpy(dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=0))


def aggregate(
    country_index: pd.DataFrame,
    gdp: pd.DataFrame,
    groups: dict[str, list[str]] = AFRICA_REGIONS,
    *,
    baseline=None,
    baseline_start: str = "2015-01",
    baseline_end: str = "2025-12",
    include_africa: bool = True,
) -> pd.DataFrame:
    """GDP-PPP-weighted regional (and, by default, Africa-wide) EPU series.

    ``country_index`` must have ``country``, ``year_month``, and ``EPU``
    columns (as produced by :func:`newsindex.index.score`); ``gdp`` must have
    ``country``, ``year_month``, and ``gdp`` columns (as produced by
    :func:`monthly_gdp`).

    Pass the :class:`newsindex.index.Baseline` used to score ``country_index``
    as ``baseline`` to reuse its window instead of repeating
    ``baseline_start``/``baseline_end``.
    """
    if baseline is not None:
        baseline_start, baseline_end = baseline.start, baseline.end
    merged = country_index.merge(gdp, on=["country", "year_month"], how="inner")

    baseline_window = merged[
        (merged["year_month"] >= baseline_start) & (merged["year_month"] <= baseline_end)
    ]
    country_stats = {
        country: _population_stats(rows["EPU"])
        for country, rows in baseline_window.groupby("country")
        if len(rows) >= 2
    }

    merged = merged[merged["country"].isin(country_stats)].copy()
    merged["m_c"] = merged["country"].map(lambda c: country_stats[c][0])
    merged["s_c"] = merged["country"].map(lambda c: country_stats[c][1])
    merged = merged[merged["s_c"] > 0]
    merged["z"] = (merged["EPU"] - merged["m_c"]) / merged["s_c"]

    all_groups = dict(groups)
    if include_africa:
        africa_members = sorted({country for members in groups.values() for country in members})
        all_groups = {**all_groups, "Africa": africa_members}

    group_rows: list[dict] = []
    for group_name, members in all_groups.items():
        group_data = merged[merged["country"].isin(members)]
        for year_month, month_rows in group_data.groupby("year_month"):
            total_gdp = month_rows["gdp"].sum()
            if total_gdp <= 0:
                continue
            weight = month_rows["gdp"] / total_gdp
            z_weighted = float((weight * month_rows["z"]).sum())
            group_rows.append(
                {
                    "group": group_name,
                    "year_month": year_month,
                    "Z": z_weighted,
                    "n_countries": int(month_rows["country"].nunique()),
                    "gdp_covered": float(total_gdp),
                }
            )

    regional = pd.DataFrame(group_rows, columns=["group", "year_month", "Z", "n_countries", "gdp_covered"])
    if regional.empty:
        return regional.rename(columns={"Z": "z"}).assign(index=pd.Series(dtype=float))

    output_rows: list[dict] = []
    for group_name, rows in regional.groupby("group"):
        window = rows[(rows["year_month"] >= baseline_start) & (rows["year_month"] <= baseline_end)]
        if len(window) < 2:
            continue
        m_r, s_r = _population_stats(window["Z"])
        if s_r <= 0:
            continue
        for _, row in rows.iterrows():
            output_rows.append(
                {
                    "group": group_name,
                    "year_month": row["year_month"],
                    "z": row["Z"],
                    "index": 100.0 + 10.0 * (row["Z"] - m_r) / s_r,
                    "n_countries": row["n_countries"],
                    "gdp_covered": row["gdp_covered"],
                }
            )

    result = pd.DataFrame(
        output_rows, columns=["group", "year_month", "z", "index", "n_countries", "gdp_covered"]
    )
    return result.sort_values(["group", "year_month"]).reset_index(drop=True)
