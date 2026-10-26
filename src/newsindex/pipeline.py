"""One-call orchestration of the EPU pipeline for non-technical users.

``run_index`` and ``run_index_with_baseline`` wrap the building blocks in
:mod:`newsindex.articles`, :mod:`newsindex.dictionary`, :mod:`newsindex.dedup`,
and :mod:`newsindex.index` into a single call: load the dictionary, read the
article archive, deduplicate, count, and compute the index. Anyone who needs
one of those steps to behave differently should call the underlying
functions directly instead -- these two exist only to remove the boilerplate
around the common case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import dedup as _dedup
from . import dictionary as _dictionary
from . import index as _index
from .articles import read_articles
from .index import Baseline


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)


def _load_dictionary(dictionary_path: str | Path, *, verbose: bool) -> _dictionary.Dictionary:
    path = Path(dictionary_path)
    if not path.exists():
        raise FileNotFoundError(f"dictionary file not found: {path}")
    _log(verbose, f"Loading dictionary from {path} ...")
    return _dictionary.load(path)


def _read_and_count(
    root: str | Path,
    dictionary_path: str | Path,
    countries: list[str] | str | None,
    *,
    verbose: bool,
) -> pd.DataFrame:
    kw = _load_dictionary(dictionary_path, verbose=verbose)

    _log(verbose, f"Reading articles from {root} ...")
    articles = read_articles(root, countries=countries)
    _log(
        verbose,
        f"Read {len(articles):,} articles across {articles['country'].nunique()} country folder(s).",
    )

    _log(verbose, "Deduplicating ...")
    deduped = _dedup.dedupe(articles)
    _log(verbose, f"{len(deduped):,} articles remain after deduplication.")

    _log(verbose, "Counting EPU articles per newspaper-month ...")
    return _index.count(deduped, kw)


def run_index(
    root: str | Path,
    dictionary_path: str | Path,
    start: str,
    end: str,
    *,
    countries: list[str] | str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read an article archive and compute a single-window EPU index in one call.

    ``root`` is the article archive directory (see :func:`newsindex.read_articles`);
    ``dictionary_path`` the EPTU-schema keyword dictionary to classify against.
    ``start``/``end`` are the inclusive ``"YYYY-MM"`` window: ``sigma``/``mu``
    are fit from it and used to score that same window -- this is the
    no-baseline variant (see :func:`newsindex.index.compute_index`).

    ``countries`` restricts which country folders under ``root`` are read
    (a list, or ``"all"``/omitted for every folder -- see
    :func:`newsindex.read_articles`). Progress is printed to stderr unless
    ``verbose=False``.
    """
    counts = _read_and_count(root, dictionary_path, countries, verbose=verbose)
    _log(verbose, f"Computing index for {start}..{end} ...")
    result = _index.compute_index(counts, start, end)
    _log(verbose, f"Done: {len(result):,} country-month rows.")
    return result


def run_index_with_baseline(
    root: str | Path,
    dictionary_path: str | Path,
    windows: list[tuple[str, str]],
    *,
    baseline: Baseline | None = None,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    countries: list[str] | str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Baseline]:
    """Read an article archive and score it against a frozen baseline in one call.

    ``root`` and ``dictionary_path`` are as in :func:`run_index`. Either pass
    an already-fitted ``baseline`` (e.g. from :meth:`newsindex.index.Baseline.load`)
    to reuse it, or pass ``baseline_start``/``baseline_end`` to fit a new one
    from this archive -- exactly one of the two must be given. ``windows`` is
    a list of inclusive ``(start, end)`` pairs scored against that baseline
    (see :func:`newsindex.index.score_windows`); this is the with-baseline
    variant.

    ``countries`` and ``verbose`` are as in :func:`run_index`.

    Returns ``(result, baseline)`` so a freshly fit baseline can be saved
    with :meth:`newsindex.index.Baseline.save` for reuse in later runs.
    """
    if baseline is None and (baseline_start is None or baseline_end is None):
        raise ValueError(
            "pass either baseline, or both baseline_start and baseline_end, to fit one"
        )
    if baseline is not None and (baseline_start is not None or baseline_end is not None):
        raise ValueError("pass baseline on its own, not alongside baseline_start/baseline_end")

    counts = _read_and_count(root, dictionary_path, countries, verbose=verbose)

    if baseline is None:
        _log(verbose, f"Fitting baseline {baseline_start}..{baseline_end} ...")
        baseline = _index.fit_baseline(counts, baseline_start, baseline_end)
    else:
        _log(verbose, f"Using provided baseline ({baseline.start}..{baseline.end}).")

    _log(verbose, f"Scoring {len(windows)} window(s) against the baseline ...")
    result = _index.score_windows(counts, baseline, windows)
    _log(verbose, f"Done: {len(result):,} country-month rows.")
    return result, baseline
