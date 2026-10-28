"""newsindex: build news-based Economic Policy Uncertainty (EPU) indices.

See the project README for the method and the module-by-module API
(``match``, ``dictionary``, ``dedup``, ``index``, ``aggregate``).
"""

from __future__ import annotations

from . import aggregate, dedup, dictionary, index, match, weights
from ._version import __version__
from .aggregate import AFRICA_REGIONS
from .articles import read_articles
from .countries import countries
from .diagnostics import coverage, dictionary_coverage, health_check, validate_archive
from .pipeline import (
    index_all,
    index_countries,
    index_region,
    index_since,
    quick_index,
    rerun,
    run_index,
    run_index_with_baseline,
)
from .plot import (
    plot,
    plot_baseline,
    plot_compare,
    plot_components,
    plot_correlation,
    plot_dashboard,
    plot_distribution,
    plot_heatmap,
    plot_rank,
    plot_regions,
    plot_rolling,
    save_plot,
    theme,
)
from .report import describe, print_period, rebase, resample, smooth
from .report import report as report
from .report import summarize, top
from .save import (
    autosave,
    load,
    save,
    save_all,
    save_csv,
    save_excel,
    save_gsheet,
    save_json,
    save_parquet,
    save_report,
    save_sql,
    save_versioned,
)

__all__ = [
    "__version__",
    "match",
    "dictionary",
    "dedup",
    "index",
    "aggregate",
    "weights",
    "read_articles",
    "countries",
    "AFRICA_REGIONS",
    "run_index",
    "run_index_with_baseline",
    "quick_index",
    "index_all",
    "index_countries",
    "index_region",
    "index_since",
    "rerun",
    "summarize",
    "print_period",
    "describe",
    "top",
    "report",
    "plot",
    "plot_compare",
    "plot_regions",
    "plot_heatmap",
    "plot_rank",
    "plot_rolling",
    "plot_distribution",
    "plot_components",
    "plot_baseline",
    "plot_correlation",
    "plot_dashboard",
    "theme",
    "save_plot",
    "resample",
    "rebase",
    "smooth",
    "coverage",
    "validate_archive",
    "dictionary_coverage",
    "health_check",
    "save",
    "save_csv",
    "save_excel",
    "save_json",
    "save_parquet",
    "save_sql",
    "save_gsheet",
    "save_all",
    "save_versioned",
    "autosave",
    "load",
    "save_report",
]
