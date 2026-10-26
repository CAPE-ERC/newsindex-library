"""newsindex: build news-based Economic Policy Uncertainty (EPU) indices.

See the project README for the method and the module-by-module API
(``match``, ``dictionary``, ``dedup``, ``index``, ``aggregate``).
"""

from __future__ import annotations

from . import aggregate, dedup, dictionary, index, match
from ._version import __version__
from .aggregate import AFRICA_REGIONS
from .articles import read_articles
from .countries import countries

__all__ = [
    "__version__",
    "match",
    "dictionary",
    "dedup",
    "index",
    "aggregate",
    "read_articles",
    "countries",
    "AFRICA_REGIONS",
]
