"""One-call charting for a result table, in a choice of backends/styles.

Operates on a result table as returned by :func:`newsindex.run_index`,
:func:`newsindex.index.score`, :func:`newsindex.index.compute_index`, or
:func:`newsindex.index.score_windows`: ``country``, ``year_month``, ``EPU``
columns (plus ``M_t``, ``mu``, ``n_sites``, optionally ``window``), unless a
function's docstring says otherwise (:func:`plot_regions`,
:func:`plot_components`, :func:`plot_baseline` take a different shape).

``matplotlib`` is this module's dependency, but only an *optional* one of
the core package -- ``import newsindex`` never requires it, only actually
calling one of these functions does (install with ``pip install
newsindex[plot]``). ``backend="plotly"``/``backend="seaborn"`` need their
own extras (``newsindex[plotly]``/``newsindex[seaborn]``) on top of that,
only when that backend is actually requested. Every import of these
libraries happens lazily inside the function that needs it, never at
import time of this module.

Functions that return a chart return the backend's native figure object:
a ``matplotlib.figure.Figure`` for ``backend="matplotlib"``/``"seaborn"``
(save with ``fig.savefig(path)``), or a ``plotly.graph_objects.Figure``
for ``backend="plotly"`` (save/show with ``fig.write_html(path)``/
``fig.show()``). :func:`save_plot` dispatches to the right one for you.
"""

from __future__ import annotations

from pathlib import Path


def _require_matplotlib(fn_name: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib installed
        raise ImportError(
            f"{fn_name}() requires matplotlib -- install with: pip install newsindex[plot]"
        ) from exc
    return plt


def _require_plotly(fn_name: str):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without plotly installed
        raise ImportError(
            f"{fn_name}() with backend='plotly' requires plotly -- install with: pip install newsindex[plotly]"
        ) from exc
    return go


def _require_seaborn(fn_name: str):
    try:
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - exercised only without seaborn installed
        raise ImportError(
            f"{fn_name}() with backend='seaborn' requires seaborn -- install with: pip install newsindex[seaborn]"
        ) from exc
    return sns


def _check_backend(backend: str, fn_name: str) -> None:
    if backend not in ("matplotlib", "plotly", "seaborn"):
        raise ValueError(f"unknown backend {backend!r} for {fn_name}(); use 'matplotlib', 'plotly', or 'seaborn'")


def plot(result, *, kind: str = "line", backend: str = "matplotlib", **kwargs):
    """One line/bar/area per country of ``EPU`` over ``year_month``, zero configuration.

    ``kind`` is one of ``"line"``, ``"bar"``, ``"area"``. ``**kwargs`` are
    passed through to the underlying plotting call. Returns a
    ``matplotlib.figure.Figure`` (``backend="matplotlib"``/``"seaborn"``)
    or a ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    if kind not in ("line", "bar", "area"):
        raise ValueError(f"unknown kind {kind!r}; use 'line', 'bar', or 'area'")
    _check_backend(backend, "plot")

    if backend == "plotly":
        go = _require_plotly("plot")
        fig = go.Figure()
        trace_kind = {"line": "lines", "bar": None, "area": "lines"}[kind]
        for country in sorted(result["country"].unique()):
            rows = result[result["country"] == country].sort_values("year_month")
            if kind == "bar":
                fig.add_bar(x=rows["year_month"], y=rows["EPU"], name=country)
            else:
                fig.add_trace(
                    go.Scatter(
                        x=rows["year_month"],
                        y=rows["EPU"],
                        mode=trace_kind,
                        name=country,
                        fill="tozeroy" if kind == "area" else None,
                    )
                )
        fig.update_layout(title="EPU by country", xaxis_title="year_month", yaxis_title="EPU")
        return fig

    plt = _require_matplotlib("plot")
    if backend == "seaborn":
        _require_seaborn("plot").set_theme()
    fig, ax = plt.subplots()
    for country in sorted(result["country"].unique()):
        rows = result[result["country"] == country].sort_values("year_month")
        if kind == "line":
            ax.plot(rows["year_month"], rows["EPU"], label=country, **kwargs)
        elif kind == "area":
            ax.fill_between(rows["year_month"], rows["EPU"], alpha=0.4, label=country, **kwargs)
        else:  # bar
            ax.bar(rows["year_month"], rows["EPU"], label=country, alpha=0.7, **kwargs)
    ax.set_xlabel("year_month")
    ax.set_ylabel("EPU")
    ax.set_title("EPU by country")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    return fig


def plot_compare(result, countries=None, *, backend: str = "matplotlib"):
    """Multi-country overlay of ``EPU`` on one chart.

    ``countries=None`` plots every country present in ``result``. Returns
    a ``matplotlib.figure.Figure`` (``backend="matplotlib"``/``"seaborn"``)
    or a ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    _check_backend(backend, "plot_compare")
    subset = result if countries is None else result[result["country"].isin(countries)]
    return plot(subset, kind="line", backend=backend)


def plot_regions(regions_result, *, backend: str = "matplotlib"):
    """One line per ``group`` of ``regions_result`` (output of
    :func:`newsindex.aggregate.aggregate`), plotting its ``index`` column
    over ``year_month``.

    Returns a ``matplotlib.figure.Figure``
    (``backend="matplotlib"``/``"seaborn"``) or a
    ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    _check_backend(backend, "plot_regions")

    if backend == "plotly":
        go = _require_plotly("plot_regions")
        fig = go.Figure()
        for group in sorted(regions_result["group"].unique()):
            rows = regions_result[regions_result["group"] == group].sort_values("year_month")
            fig.add_trace(go.Scatter(x=rows["year_month"], y=rows["index"], mode="lines", name=group))
        fig.update_layout(title="Regional EPU index", xaxis_title="year_month", yaxis_title="index")
        return fig

    plt = _require_matplotlib("plot_regions")
    if backend == "seaborn":
        _require_seaborn("plot_regions").set_theme()
    fig, ax = plt.subplots()
    for group in sorted(regions_result["group"].unique()):
        rows = regions_result[regions_result["group"] == group].sort_values("year_month")
        ax.plot(rows["year_month"], rows["index"], label=group)
    ax.set_xlabel("year_month")
    ax.set_ylabel("index")
    ax.set_title("Regional EPU index")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    return fig


def plot_heatmap(result, *, backend: str = "matplotlib"):
    """Country x month heatmap of ``EPU`` intensity.

    Returns a ``matplotlib.figure.Figure``
    (``backend="matplotlib"``/``"seaborn"``) or a
    ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    _check_backend(backend, "plot_heatmap")
    pivot = result.pivot_table(index="country", columns="year_month", values="EPU", aggfunc="mean")
    pivot = pivot.sort_index()[sorted(pivot.columns)]

    if backend == "plotly":
        go = _require_plotly("plot_heatmap")
        fig = go.Figure(
            data=go.Heatmap(z=pivot.to_numpy(), x=list(pivot.columns), y=list(pivot.index), colorscale="Viridis")
        )
        fig.update_layout(title="EPU heatmap", xaxis_title="year_month", yaxis_title="country")
        return fig

    plt = _require_matplotlib("plot_heatmap")
    if backend == "seaborn":
        sns = _require_seaborn("plot_heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(pivot, ax=ax, cmap="viridis")
        ax.set_title("EPU heatmap")
        return fig

    fig, ax = plt.subplots()
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_title("EPU heatmap")
    fig.colorbar(im, ax=ax, label="EPU")
    return fig


def plot_rank(result, year_month: str, *, backend: str = "matplotlib"):
    """Bar chart ranking countries by ``EPU`` for one ``year_month``.

    Raises :class:`ValueError` if no rows match ``year_month``. Returns a
    ``matplotlib.figure.Figure`` (``backend="matplotlib"``/``"seaborn"``)
    or a ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    _check_backend(backend, "plot_rank")
    rows = result[result["year_month"] == year_month].sort_values("EPU", ascending=False)
    if rows.empty:
        raise ValueError(f"no rows with year_month == {year_month!r}")

    if backend == "plotly":
        go = _require_plotly("plot_rank")
        fig = go.Figure(go.Bar(x=rows["country"], y=rows["EPU"]))
        fig.update_layout(title=f"EPU ranking, {year_month}", xaxis_title="country", yaxis_title="EPU")
        return fig

    plt = _require_matplotlib("plot_rank")
    if backend == "seaborn":
        _require_seaborn("plot_rank").set_theme()
    fig, ax = plt.subplots()
    ax.bar(rows["country"], rows["EPU"])
    ax.set_xlabel("country")
    ax.set_ylabel("EPU")
    ax.set_title(f"EPU ranking, {year_month}")
    fig.autofmt_xdate(rotation=45)
    return fig


def _rolling_epu(result, window: int):
    sorted_result = result.sort_values(["country", "year_month"])
    smoothed = sorted_result.groupby("country")["EPU"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    return sorted_result.assign(EPU_smoothed=smoothed)


def plot_rolling(result, window: int = 3, *, backend: str = "matplotlib"):
    """Smoothed (rolling-average) trend line over the raw ``EPU`` series.

    The rolling average is computed inline here (per-country, sorted by
    ``year_month``, ``min_periods=1``). Returns a
    ``matplotlib.figure.Figure`` (``backend="matplotlib"``/``"seaborn"``)
    or a ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    _check_backend(backend, "plot_rolling")
    smoothed = _rolling_epu(result, window)

    if backend == "plotly":
        go = _require_plotly("plot_rolling")
        fig = go.Figure()
        for country in sorted(smoothed["country"].unique()):
            rows = smoothed[smoothed["country"] == country]
            fig.add_trace(go.Scatter(x=rows["year_month"], y=rows["EPU"], mode="lines", name=f"{country} (raw)", opacity=0.3))
            fig.add_trace(go.Scatter(x=rows["year_month"], y=rows["EPU_smoothed"], mode="lines", name=f"{country} (smoothed)"))
        fig.update_layout(title=f"EPU, {window}-month rolling average", xaxis_title="year_month", yaxis_title="EPU")
        return fig

    plt = _require_matplotlib("plot_rolling")
    if backend == "seaborn":
        _require_seaborn("plot_rolling").set_theme()
    fig, ax = plt.subplots()
    for country in sorted(smoothed["country"].unique()):
        rows = smoothed[smoothed["country"] == country]
        ax.plot(rows["year_month"], rows["EPU"], alpha=0.3, linestyle="--")
        ax.plot(rows["year_month"], rows["EPU_smoothed"], label=country)
    ax.set_xlabel("year_month")
    ax.set_ylabel("EPU")
    ax.set_title(f"EPU, {window}-month rolling average")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    return fig


def plot_distribution(result, *, kind: str = "hist", backend: str = "matplotlib"):
    """Histogram (``kind="hist"``) or boxplot (``kind="box"``) of ``EPU``
    values across countries/months.

    Returns a ``matplotlib.figure.Figure``
    (``backend="matplotlib"``/``"seaborn"``) or a
    ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    if kind not in ("hist", "box"):
        raise ValueError(f"unknown kind {kind!r}; use 'hist' or 'box'")
    _check_backend(backend, "plot_distribution")

    if backend == "plotly":
        go = _require_plotly("plot_distribution")
        if kind == "hist":
            fig = go.Figure(go.Histogram(x=result["EPU"]))
        else:
            fig = go.Figure(go.Box(y=result["EPU"]))
        fig.update_layout(title=f"EPU distribution ({kind})")
        return fig

    plt = _require_matplotlib("plot_distribution")
    if backend == "seaborn":
        sns = _require_seaborn("plot_distribution")
        fig, ax = plt.subplots()
        if kind == "hist":
            sns.histplot(result["EPU"], ax=ax)
        else:
            sns.boxplot(y=result["EPU"], ax=ax)
        ax.set_title(f"EPU distribution ({kind})")
        return fig

    fig, ax = plt.subplots()
    if kind == "hist":
        ax.hist(result["EPU"], bins=20)
        ax.set_xlabel("EPU")
        ax.set_ylabel("count")
    else:
        ax.boxplot(result["EPU"].dropna(), tick_labels=["EPU"])
    ax.set_title(f"EPU distribution ({kind})")
    return fig


def plot_components(counts, *, backend: str = "matplotlib"):
    """Stacked area of EPU vs. non-EPU article volume over time.

    ``counts`` is a table like :func:`newsindex.index.count`'s output:
    ``country``, ``site``, ``year_month``, ``epu_articles``,
    ``all_articles``. Summed across country/site per ``year_month``.
    Returns a ``matplotlib.figure.Figure``
    (``backend="matplotlib"``/``"seaborn"``) or a
    ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    _check_backend(backend, "plot_components")
    by_month = counts.groupby("year_month", as_index=False)[["epu_articles", "all_articles"]].sum()
    by_month = by_month.sort_values("year_month")
    by_month["non_epu_articles"] = by_month["all_articles"] - by_month["epu_articles"]

    if backend == "plotly":
        go = _require_plotly("plot_components")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=by_month["year_month"], y=by_month["epu_articles"], mode="lines", name="EPU", stackgroup="one"))
        fig.add_trace(go.Scatter(x=by_month["year_month"], y=by_month["non_epu_articles"], mode="lines", name="non-EPU", stackgroup="one"))
        fig.update_layout(title="Article classification volume", xaxis_title="year_month", yaxis_title="articles")
        return fig

    plt = _require_matplotlib("plot_components")
    if backend == "seaborn":
        _require_seaborn("plot_components").set_theme()
    fig, ax = plt.subplots()
    ax.stackplot(
        by_month["year_month"],
        by_month["epu_articles"],
        by_month["non_epu_articles"],
        labels=["EPU", "non-EPU"],
    )
    ax.set_xlabel("year_month")
    ax.set_ylabel("articles")
    ax.set_title("Article classification volume")
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=45)
    return fig


def plot_baseline(baseline, *, backend: str = "matplotlib"):
    """Bar chart of a frozen :class:`newsindex.index.Baseline`'s per-site
    ``sigma`` values, grouped by country -- useful for spotting a baseline
    that's gone stale (an oddly large or near-zero sigma).

    Returns a ``matplotlib.figure.Figure``
    (``backend="matplotlib"``/``"seaborn"``) or a
    ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    _check_backend(backend, "plot_baseline")
    labels, values, countries = [], [], []
    for country in sorted(baseline.sigma):
        for site in baseline.retained_sites(country):
            labels.append(f"{country}: {site}")
            values.append(baseline.sigma[country][site])
            countries.append(country)

    if backend == "plotly":
        go = _require_plotly("plot_baseline")
        fig = go.Figure(go.Bar(x=labels, y=values))
        fig.update_layout(title=f"Baseline sigma ({baseline.start}..{baseline.end})", xaxis_title="country: site", yaxis_title="sigma")
        return fig

    plt = _require_matplotlib("plot_baseline")
    if backend == "seaborn":
        _require_seaborn("plot_baseline").set_theme()
    fig, ax = plt.subplots()
    unique_countries = sorted(set(countries))
    colors = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(unique_countries)}
    ax.bar(labels, values, color=[colors[c] for c in countries])
    ax.set_xlabel("country: site")
    ax.set_ylabel("sigma")
    ax.set_title(f"Baseline sigma ({baseline.start}..{baseline.end})")
    fig.autofmt_xdate(rotation=90)
    return fig


def plot_correlation(result, external_series, *, backend: str = "matplotlib"):
    """Dual-axis overlay of a single country's ``EPU`` against an external series.

    ``result`` must cover exactly one country (filter it first) --
    :class:`ValueError` otherwise. ``external_series`` is a
    ``pandas.Series`` indexed by ``year_month`` strings, or a two-column
    DataFrame (first column treated as the ``year_month``-like index,
    second as the value). Returns a ``matplotlib.figure.Figure``
    (``backend="matplotlib"``/``"seaborn"``) or a
    ``plotly.graph_objects.Figure`` (``backend="plotly"``).
    """
    _check_backend(backend, "plot_correlation")
    countries = result["country"].unique()
    if len(countries) != 1:
        raise ValueError(
            f"plot_correlation() needs a single-country result, got {len(countries)}: "
            f"{sorted(countries)} -- filter result to one country first"
        )
    country = countries[0]
    rows = result.sort_values("year_month")

    if hasattr(external_series, "columns"):
        ext_index = external_series.iloc[:, 0]
        ext_values = external_series.iloc[:, 1]
    else:
        ext_index = external_series.index
        ext_values = external_series.values

    if backend == "plotly":
        go = _require_plotly("plot_correlation")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rows["year_month"], y=rows["EPU"], mode="lines", name=f"{country} EPU", yaxis="y1"))
        fig.add_trace(go.Scatter(x=list(ext_index), y=list(ext_values), mode="lines", name="external", yaxis="y2"))
        fig.update_layout(
            title=f"{country} EPU vs. external series",
            xaxis_title="year_month",
            yaxis=dict(title="EPU"),
            yaxis2=dict(title="external", overlaying="y", side="right"),
        )
        return fig

    plt = _require_matplotlib("plot_correlation")
    if backend == "seaborn":
        _require_seaborn("plot_correlation").set_theme()
    fig, ax1 = plt.subplots()
    ax1.plot(rows["year_month"], rows["EPU"], color="tab:blue", label=f"{country} EPU")
    ax1.set_xlabel("year_month")
    ax1.set_ylabel("EPU", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(list(ext_index), list(ext_values), color="tab:orange", label="external")
    ax2.set_ylabel("external", color="tab:orange")
    ax1.set_title(f"{country} EPU vs. external series")
    fig.autofmt_xdate(rotation=45)
    return fig


def plot_dashboard(result, *, backend: str = "matplotlib"):
    """One multi-panel figure: trend, latest-month ranking, and EPU distribution.

    This function only supports ``backend="matplotlib"`` -- a composed
    multi-panel figure isn't implemented for plotly/seaborn.
    :class:`NotImplementedError` is raised for any other ``backend``.
    Returns a ``matplotlib.figure.Figure``.
    """
    if backend != "matplotlib":
        raise NotImplementedError(
            f"plot_dashboard() only supports backend='matplotlib', got {backend!r}"
        )
    plt = _require_matplotlib("plot_dashboard")

    latest_month = sorted(result["year_month"].unique())[-1]
    rank_rows = result[result["year_month"] == latest_month].sort_values("EPU", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    trend_ax = axes[0]
    for country in sorted(result["country"].unique()):
        rows = result[result["country"] == country].sort_values("year_month")
        trend_ax.plot(rows["year_month"], rows["EPU"], label=country)
    trend_ax.set_title("Trend")
    trend_ax.set_xlabel("year_month")
    trend_ax.set_ylabel("EPU")
    trend_ax.legend(fontsize="small")
    trend_ax.tick_params(axis="x", rotation=45)

    rank_ax = axes[1]
    rank_ax.bar(rank_rows["country"], rank_rows["EPU"])
    rank_ax.set_title(f"Ranking, {latest_month}")
    rank_ax.tick_params(axis="x", rotation=45)

    dist_ax = axes[2]
    dist_ax.hist(result["EPU"], bins=20)
    dist_ax.set_title("Distribution")
    dist_ax.set_xlabel("EPU")

    fig.tight_layout()
    return fig


def theme(style: str = "default", *, backend: str = "matplotlib") -> None:
    """Apply a consistent default visual style for subsequent charts.

    Mutates GLOBAL matplotlib state (``matplotlib.rcParams``) -- call this
    once near the top of a script, before creating any figures. It affects
    every matplotlib figure created afterward in the process, not just
    ones made by :mod:`newsindex.plot`. Only ``backend="matplotlib"`` is
    meaningful here (seaborn/plotly have their own separate styling
    mechanisms); ``backend`` is accepted for a consistent call signature
    with the rest of this module but any value other than
    ``"matplotlib"`` raises :class:`NotImplementedError`.
    """
    if backend != "matplotlib":
        raise NotImplementedError(f"theme() only supports backend='matplotlib', got {backend!r}")
    if style != "default":
        raise ValueError(f"unknown style {style!r}; only 'default' is available")

    plt = _require_matplotlib("theme")
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(
                color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
            ),
            "font.family": "sans-serif",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_plot(fig, path, **kwargs) -> None:
    """Save a chart produced by this module to ``path``.

    Dispatches on ``fig``'s type: a matplotlib ``Figure`` (has
    ``.savefig``) is saved with ``fig.savefig(path, **kwargs)``; a plotly
    ``Figure`` (has ``.write_html``/``.write_image``) is saved with
    ``fig.write_html(path, **kwargs)`` when ``path`` ends in ``.html``
    (needs no extra dependency beyond plotly itself), or
    ``fig.write_image(path, **kwargs)`` otherwise (needs the ``kaleido``
    package for static image export -- ``pip install kaleido``).
    :class:`TypeError` is raised for any other object.
    """
    path = Path(path)
    if hasattr(fig, "savefig"):
        fig.savefig(path, **kwargs)
    elif hasattr(fig, "write_html") and hasattr(fig, "write_image"):
        if path.suffix.lower() == ".html":
            fig.write_html(path, **kwargs)
        else:
            try:
                fig.write_image(path, **kwargs)
            except ValueError as exc:
                raise ImportError(
                    "save_plot() to a non-.html path for a plotly figure requires kaleido -- "
                    "install with: pip install kaleido"
                ) from exc
    else:
        raise TypeError(
            f"save_plot() doesn't know how to save a {type(fig).__name__!r} object -- "
            "expected a matplotlib or plotly Figure"
        )
