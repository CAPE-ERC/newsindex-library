"""Turn a result table into something a reader can act on without opening pandas.

Operates on a result table as returned by :func:`newsindex.run_index`,
:func:`newsindex.index.score`, :func:`newsindex.index.compute_index`, or
:func:`newsindex.index.score_windows`: ``country``, ``year_month``, ``EPU``,
``M_t``, ``mu``, ``n_sites`` columns (optionally ``window``).

Plain result DataFrames aren't given a custom ``__repr__`` here (that would
require subclassing ``pandas.DataFrame``, extra machinery this library has
deliberately avoided) -- :func:`summarize`, :func:`describe`, and
:func:`print_period` are the "nicer view" instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _biggest_move(rows: pd.DataFrame) -> tuple[float, str] | tuple[None, None]:
    """Largest absolute month-over-month EPU change within one country's rows.

    ``rows`` must already be sorted by ``year_month``. Returns
    ``(abs_change, later_year_month)``, or ``(None, None)`` if fewer than
    two months.
    """
    if len(rows) < 2:
        return None, None
    diffs = rows["EPU"].diff().abs()
    idx = diffs.idxmax()
    return float(diffs.loc[idx]), str(rows.loc[idx, "year_month"])


def summarize(result: pd.DataFrame) -> str:
    """Plain-English multi-line summary, one block per country.

    For each country: first and last EPU value+month, percent change
    first-to-last, and the single month with the largest month-over-month
    absolute EPU change. A country with fewer than two months reports only
    its single value.
    """
    lines: list[str] = []
    for country in sorted(result["country"].unique()):
        rows = result[result["country"] == country].sort_values("year_month")
        first_month, first_value = str(rows.iloc[0]["year_month"]), float(rows.iloc[0]["EPU"])
        if len(rows) < 2:
            lines.append(f"{country}: EPU was {first_value:.1f} in {first_month} (only month available).")
            continue

        last_month, last_value = str(rows.iloc[-1]["year_month"]), float(rows.iloc[-1]["EPU"])
        pct_change = ((last_value - first_value) / first_value * 100.0) if first_value != 0 else float("nan")
        direction = "rose" if last_value >= first_value else "fell"

        move_value, move_month = _biggest_move(rows)

        line = (
            f"{country}: EPU {direction} from {first_value:.1f} ({first_month}) to "
            f"{last_value:.1f} ({last_month}), {pct_change:+.1f}%."
        )
        if move_month is not None:
            line += f" Largest month-over-month move: {move_value:.1f} points, in {move_month}."
        lines.append(line)

    return "\n".join(lines)


def print_period(result: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Pretty-print the slice of ``result`` with ``year_month`` in ``[start, end]``.

    Prints the full, untruncated table to stdout and also returns it, so it
    can be used programmatically. Raises :class:`ValueError` if the window
    matches no rows (most likely a typo'd ``start``/``end``) instead of
    silently printing nothing.
    """
    window = result[(result["year_month"] >= start) & (result["year_month"] <= end)]
    if window.empty:
        raise ValueError(f"no rows with year_month in [{start}, {end}] -- check for a typo in the window")

    with pd.option_context("display.max_rows", None, "display.width", None):
        print(window.to_string(index=False))
    return window


def describe(result: pd.DataFrame) -> pd.DataFrame:
    """Per-country min/max/mean/std of EPU, plus the biggest month-over-month move.

    One row per country: ``country``, ``min``, ``max``, ``mean``, ``std``,
    ``n_months``, ``biggest_move`` (largest absolute month-over-month EPU
    change; ``NaN`` if fewer than two months), ``biggest_move_month`` (the
    later month of that pair, ``None`` if not applicable).
    """
    rows: list[dict] = []
    for country in sorted(result["country"].unique()):
        country_rows = result[result["country"] == country].sort_values("year_month")
        epu = country_rows["EPU"]
        move_value, move_month = _biggest_move(country_rows)
        rows.append(
            {
                "country": country,
                "min": float(epu.min()),
                "max": float(epu.max()),
                "mean": float(epu.mean()),
                "std": float(epu.std(ddof=0)) if len(epu) > 1 else float("nan"),
                "n_months": int(len(country_rows)),
                "biggest_move": move_value if move_value is not None else float("nan"),
                "biggest_move_month": move_month,
            }
        )
    columns = ["country", "min", "max", "mean", "std", "n_months", "biggest_move", "biggest_move_month"]
    return pd.DataFrame(rows, columns=columns)


def top(result: pd.DataFrame, n: int = 5, by: str = "EPU") -> pd.DataFrame:
    """Top-``n`` rows of ``result``, sorted descending by column ``by``."""
    if by not in result.columns:
        raise ValueError(f"{by!r} is not a column of result; available columns: {list(result.columns)}")
    return result.sort_values(by, ascending=False, kind="stable").head(n).reset_index(drop=True)


def _markdown_table(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


_RESAMPLE_FREQS = {"Q": "Q", "A": "Y", "Y": "Y"}


def resample(result: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    """Reshape ``result`` from monthly to quarterly or annual averages.

    ``freq`` is ``"Q"`` (quarterly) or ``"A"``/``"Y"`` (annual). Per
    country, ``year_month`` ("YYYY-MM") is parsed to a monthly
    :class:`pandas.Period`, converted to the target frequency, and ``EPU``
    (plus ``M_t``/``n_sites`` where present) are averaged within each
    resulting period. Output columns: ``country``, ``period`` (e.g.
    ``"2024Q1"`` or ``"2024"``), and the averaged numeric columns.
    """
    target = _RESAMPLE_FREQS.get(freq.upper())
    if target is None:
        raise ValueError(f"unsupported freq {freq!r}; use 'Q', 'A', or 'Y'")

    working = result.copy()
    working["period"] = pd.PeriodIndex(working["year_month"], freq="M").asfreq(target).astype(str)

    value_cols = [c for c in ("EPU", "M_t", "n_sites") if c in working.columns]
    grouped = working.groupby(["country", "period"], as_index=False)[value_cols].mean()
    return grouped.sort_values(["country", "period"]).reset_index(drop=True)


def rebase(result: pd.DataFrame, new_baseline_period: tuple[str, str]) -> pd.DataFrame:
    """Recompute the 100-mean rescaling of ``EPU`` against a different reference period.

    Works directly off the already-scored ``EPU`` column -- no ``M_t``/
    ``mu``/:class:`newsindex.index.Baseline` refit needed. Per country,
    ``factor = 100 / mean(EPU in new_baseline_period)``, and every EPU
    value for that country is multiplied by its own factor. Returns a copy
    of ``result`` with ``EPU`` replaced; all other columns are unchanged.

    Raises :class:`ValueError` (naming the country) if a country has no
    rows in ``new_baseline_period``, or a zero mean there.
    """
    start, end = new_baseline_period
    working = result.copy()
    rebased = working["EPU"].astype(float).copy()

    for country, rows in working.groupby("country"):
        window = rows[(rows["year_month"] >= start) & (rows["year_month"] <= end)]
        if window.empty:
            raise ValueError(f"{country!r} has no rows in new_baseline_period [{start}, {end}]")
        baseline_mean = float(window["EPU"].mean())
        if baseline_mean == 0:
            raise ValueError(f"{country!r} has a zero mean EPU in new_baseline_period [{start}, {end}] -- cannot rebase")
        factor = 100.0 / baseline_mean
        rebased.loc[rows.index] = rows["EPU"].astype(float) * factor

    working["EPU"] = rebased
    return working


def smooth(result: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add an ``EPU_smoothed`` rolling-average column, per country.

    Returns a copy of ``result`` with the raw ``EPU`` column unchanged and
    a new ``EPU_smoothed`` column: the trailing ``window``-month average,
    sorted by ``year_month`` within each country. ``min_periods=1`` means
    the first ``window - 1`` months of each country are averaged over
    however many months are available so far, rather than dropped.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    working = result.sort_values(["country", "year_month"]).copy()
    working["EPU_smoothed"] = working.groupby("country")["EPU"].transform(
        lambda s: s.rolling(window=window, min_periods=1).mean()
    )
    return working.sort_index()


def report(result: pd.DataFrame, path: str | Path) -> None:
    """Render a minimal one-page report to ``path``.

    Format is chosen by ``path``'s extension: ``.html`` for an HTML page,
    ``.md``/``.markdown`` for Markdown. Any other extension raises
    :class:`ValueError`. The report is a title, the :func:`summarize` text,
    and the :func:`describe` table -- a readable static page, not a styled
    dashboard.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    summary_text = summarize(result)
    stats = describe(result)

    if suffix == ".html":
        summary_html = "\n".join(f"<p>{line}</p>" for line in summary_text.splitlines())
        content = (
            "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>newsindex report</title></head>\n"
            f"<body>\n<h1>newsindex EPU report</h1>\n{summary_html}\n"
            f"<h2>Summary statistics</h2>\n{stats.to_html(index=False)}\n</body></html>\n"
        )
    elif suffix in (".md", ".markdown"):
        summary_md = "\n\n".join(summary_text.splitlines())
        content = (
            f"# newsindex EPU report\n\n{summary_md}\n\n## Summary statistics\n\n{_markdown_table(stats)}\n"
        )
    else:
        raise ValueError(f"unsupported report extension {suffix!r}; use .html, .md, or .markdown")

    path.write_text(content, encoding="utf-8")
