"""Ready-made weight sources for :func:`newsindex.aggregate.aggregate`.

Each of these matches ``aggregate()``'s existing ``weights`` parameter
contract directly: either a DataFrame with ``country``/``year_month``/a
value column, or a callable ``(month_rows, group_name) -> pandas.Series``
aligned to ``month_rows.index``. None of these add a new weighting
*mechanism* -- they're just common cases that would otherwise mean writing
the same small table or lambda by hand every time.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregate as _aggregate


def equal(month_rows: pd.DataFrame, group_name: str) -> pd.Series:
    """Equal weighting: every country in the group counts the same.

    Matches ``aggregate()``'s callable-weights signature directly -- pass
    ``newsindex.weights.equal`` itself as ``weights``, no call needed.
    """
    return pd.Series(1.0, index=month_rows.index)


def gdp(path: str | Path | pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Thin wrapper around :func:`newsindex.aggregate.monthly_gdp`."""
    return _aggregate.monthly_gdp(path, **kwargs)


def population(
    path: str | Path | pd.DataFrame,
    *,
    country_col: str = "Country",
    year_col: str = "Year",
    value_col: str = "Population",
    scale: float = 1.0,
    end: str | None = None,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Interpolate annual population (or any other annual-by-country metric)
    to monthly values, producing a ``population`` column.

    Same interpolation behaviour as :func:`newsindex.aggregate.monthly_gdp`
    (log-linear through each pair of annual figures, flat-carried forward
    after the last one) -- just a different output column name and default
    source-column names suited to a population file.
    """
    return _aggregate._interpolate_annual_to_monthly(
        path,
        country_col=country_col,
        year_col=year_col,
        value_col=value_col,
        scale=scale,
        end=end,
        aliases=aliases,
        output_col="population",
    )


def custom(column: str) -> Callable[[pd.DataFrame, str], pd.Series]:
    """Weight by an existing column of ``country_index`` itself.

    For the case where the weighting value (e.g. a prior index level, an
    already-merged column) is already present on ``country_index`` passed
    into ``aggregate()`` -- callable weights see ``country_index`` as-is,
    unmerged, so this works only if ``column`` is already there.
    """
    return lambda month_rows, group_name: month_rows[column]


def pca(
    table: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    country_col: str = "country",
    month_col: str = "year_month",
    verbose: bool = False,
) -> pd.DataFrame:
    """Derive a single composite weight from the first principal component
    across several candidate indicator columns.

    ``table`` has one row per country-month and several numeric indicator
    columns (e.g. ``gdp``, ``population``, ``trade_volume``), typically
    built by merging several of this module's other helpers together first.
    ``columns=None`` uses every column of ``table`` except ``country_col``/
    ``month_col``.

    Each indicator is standardised to zero mean, unit standard deviation
    across all rows, then the first principal component's loadings (via
    eigendecomposition of the covariance matrix) combine them into one
    composite score per row -- the first PC captures the dominant shared
    variation among the indicators, a standard technique in composite-index
    construction. The component's sign is fixed so its largest-magnitude
    loading is positive (otherwise PCA's sign is arbitrary), and the
    composite is shifted so its minimum is zero, since ``aggregate()``
    requires non-negative weights.

    Returns a DataFrame with ``country_col``, ``month_col``, and a
    ``pca_weight`` column -- usable as ``weight_col="pca_weight"`` on
    ``aggregate()``.
    """
    if columns is None:
        columns = [c for c in table.columns if c not in (country_col, month_col)]
    if len(columns) < 2:
        raise ValueError(
            f"need at least 2 indicator columns for PCA, got {columns!r}; "
            "pass columns=[...] to pick which ones, or add more indicators to table"
        )
    if len(table) < 3:
        raise ValueError(f"need at least 3 rows for PCA, got {len(table)}")

    data = table[columns].to_numpy(dtype=float)
    means = data.mean(axis=0)
    stds = data.std(axis=0, ddof=0)
    if np.any(stds == 0):
        constant = [columns[i] for i in range(len(columns)) if stds[i] == 0]
        raise ValueError(f"indicator column(s) {constant} are constant across table -- cannot standardise")
    standardized = (data - means) / stds

    cov = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    loadings = eigenvectors[:, -1]  # eigh returns ascending eigenvalues; last column is the largest
    if loadings[np.argmax(np.abs(loadings))] < 0:
        loadings = -loadings

    composite = standardized @ loadings
    composite = composite - composite.min()
    if composite.max() == 0:
        raise ValueError("composite PCA score is constant across all rows -- cannot use as a differentiating weight")

    if verbose:
        parts = ", ".join(f"{col}={loading:.2f}" for col, loading in zip(columns, loadings))
        print(f"PCA weights: {parts}", file=sys.stderr)

    return table[[country_col, month_col]].assign(pca_weight=composite)
