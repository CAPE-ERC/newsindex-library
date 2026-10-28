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
from .report import describe, print_period, rebase, resample, smooth
from .report import report as report
from .report import summarize, top

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
    "resample",
    "rebase",
    "smooth",
    "coverage",
    "validate_archive",
    "dictionary_coverage",
    "health_check",
]
