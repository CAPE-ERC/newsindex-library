"""One-call orchestration of the EPU pipeline for non-technical users.

``run_index`` and ``run_index_with_baseline`` wrap the building blocks in
:mod:`newsindex.dictionary` and :mod:`newsindex.index` into a single call:
load the dictionary, stream the article archive straight into EPU counts
(country-month by country-month, via :func:`newsindex.index.count_archive`,
never materializing the whole requested scope in memory at once), and
compute the index. Anyone who needs read/dedupe/count as separate,
in-memory steps should call :func:`newsindex.read_articles`,
:func:`newsindex.dedup.dedupe`, and :func:`newsindex.index.count` directly
instead -- these two exist only to remove the boilerplate around the
common case.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd

from . import aggregate as _aggregate
from . import dictionary as _dictionary
from . import index as _index
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
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    kw = _load_dictionary(dictionary_path, verbose=verbose)

    _log(verbose, f"Streaming articles from {root} (one country-month at a time) ...")
    progress = (lambda message: _log(True, f"  {message}")) if verbose else None
    counts = _index.count_archive(
        root, kw, countries=countries, progress=progress, start=start, end=end
    )
    _log(
        verbose,
        f"Counted {int(counts['all_articles'].sum()):,} articles "
        f"({int(counts['epu_articles'].sum()):,} EPU) across {counts['country'].nunique()} countr(y/ies).",
    )
    return counts


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
    no-baseline variant (see :func:`newsindex.index.compute_index`). Only
    this window's month folders are ever read -- no earlier archive
    history is streamed, scanned, or shown in progress output.

    ``countries`` restricts which country folders under ``root`` are read
    (a list, or ``"all"``/omitted for every folder -- see
    :func:`newsindex.read_articles`). Progress is printed to stderr unless
    ``verbose=False``.
    """
    counts = _read_and_count(root, dictionary_path, countries, verbose=verbose, start=start, end=end)
    _log(verbose, f"Computing index for {start}..{end} ...")
    result = _index.compute_index(counts, start, end)
    _log(verbose, f"Done: {len(result):,} country-month rows.")
    result.attrs["newsindex_config"] = {
        "fn": "run_index",
        "root": root,
        "dictionary_path": dictionary_path,
        "start": start,
        "end": end,
        "countries": countries,
        "verbose": verbose,
    }
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

    fitted_fresh = baseline is None
    if fitted_fresh:
        _log(verbose, f"Fitting baseline {baseline_start}..{baseline_end} ...")
        baseline = _index.fit_baseline(counts, baseline_start, baseline_end)
    else:
        _log(verbose, f"Using provided baseline ({baseline.start}..{baseline.end}).")

    _log(verbose, f"Scoring {len(windows)} window(s) against the baseline ...")
    result = _index.score_windows(counts, baseline, windows)
    _log(verbose, f"Done: {len(result):,} country-month rows.")

    config = {
        "fn": "run_index_with_baseline",
        "root": root,
        "dictionary_path": dictionary_path,
        "windows": windows,
        "countries": countries,
        "verbose": verbose,
    }
    if fitted_fresh:
        config["baseline_start"] = baseline_start
        config["baseline_end"] = baseline_end
    else:
        config["baseline"] = baseline
    result.attrs["newsindex_config"] = config

    return result, baseline


def quick_index(
    root: str | Path,
    dictionary_path: str | Path,
    country: str,
    year: int,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Single-country, single-calendar-year index, no window math required.

    Equivalent to ``run_index(root, dictionary_path, start=f"{year}-01",
    end=f"{year}-12", countries=[country], verbose=verbose)``.
    """
    return run_index(
        root, dictionary_path, f"{year}-01", f"{year}-12", countries=[country], verbose=verbose
    )


def index_all(
    root: str | Path,
    dictionary_path: str | Path,
    start: str,
    end: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Index every country folder under ``root``.

    An explicit, discoverably-named alias for ``run_index(root,
    dictionary_path, start, end, countries=None, verbose=verbose)`` -- the
    call reads as intent ("all countries") rather than a parameter value.
    """
    return run_index(root, dictionary_path, start, end, countries=None, verbose=verbose)


def index_countries(
    root: str | Path,
    dictionary_path: str | Path,
    countries: list[str],
    start: str,
    end: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Index a specific list of countries.

    An alias for ``run_index(root, dictionary_path, start, end,
    countries=countries, verbose=verbose)`` -- ``run_index`` already accepts
    ``countries`` as a list; this is a more discoverable, explicitly-named
    entry point for that case.
    """
    return run_index(root, dictionary_path, start, end, countries=countries, verbose=verbose)


def index_region(
    root: str | Path,
    dictionary_path: str | Path,
    region: str | list[str],
    start: str,
    end: str,
    *,
    groups: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Index a whole predefined region, without listing its member countries by hand.

    ``region`` is either a key into ``groups`` (default
    :data:`newsindex.aggregate.AFRICA_REGIONS`, e.g. ``"West"``) or an
    explicit list of country names.

    Unlike ``countries=[...]`` on ``run_index`` itself (which errors on any
    name with no matching folder -- a typo guard for a list you typed by
    hand), a region's members are only intersected with whatever country
    folders actually exist under ``root``: a region is a derived list, and
    an archive naturally won't have every member of a region scraped. Only
    raises if *none* of the region's members are present.
    """
    if groups is None:
        groups = _aggregate.AFRICA_REGIONS
    if isinstance(region, str):
        if region not in groups:
            raise ValueError(f"unknown region {region!r}; valid regions: {sorted(groups)}")
        members = groups[region]
    else:
        members = list(region)

    available = {p.name.lower(): p.name for p in Path(root).iterdir() if p.is_dir()}
    present = [available[name.lower()] for name in members if name.lower() in available]
    if not present:
        raise ValueError(
            f"none of region {region!r}'s members ({members}) have a folder under {root}; "
            f"available: {sorted(available.values())}"
        )

    return run_index(root, dictionary_path, start, end, countries=present, verbose=verbose)


def index_since(
    root: str | Path,
    dictionary_path: str | Path,
    months_back: int = 12,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Index a relative window ending at the current calendar month.

    "Now" is the machine's system clock at call time (``datetime.date.today()``),
    not anything about the archive's actual latest data -- if the archive
    doesn't yet have articles for the current month, that month's row will
    simply be absent/zero in the result, the same as any other empty month.

    The window covers exactly ``months_back`` months inclusive, e.g.
    ``months_back=12`` with today in 2026-03 covers 2025-04..2026-03.
    """
    today = datetime.date.today()
    end = f"{today.year:04d}-{today.month:02d}"

    total_months = today.year * 12 + (today.month - 1) - (months_back - 1)
    start_year, start_month = divmod(total_months, 12)
    start = f"{start_year:04d}-{start_month + 1:02d}"

    return run_index(root, dictionary_path, start, end, verbose=verbose)


def rerun(previous_result_or_config, *, verbose: bool | None = None):
    """Re-run the exact call that produced a saved result.

    ``run_index``, ``run_index_with_baseline``, and every quick-builder
    above (``quick_index``, ``index_all``, ``index_countries``,
    ``index_region``, ``index_since``) attach a config dict to their result
    via ``result.attrs["newsindex_config"]`` (``pandas.DataFrame.attrs``, an
    in-memory dict pandas carries on the object) describing how to
    reproduce that specific call. Pass that result straight back in to
    re-run it -- useful for a monthly refresh of the same request.

    ``previous_result_or_config`` is either such a result (or, for
    ``run_index_with_baseline``, its first returned element) or an
    equivalent config dict built/reloaded by hand. If ``verbose`` is given
    here it overrides the stored value; otherwise the original call's
    ``verbose`` is reused.

    A plain DataFrame reloaded from CSV (or otherwise missing this config,
    e.g. because it was copied with ``.copy()`` before a pandas version
    that preserves ``attrs`` through that) has no usable config --
    ``attrs`` only exists in memory and does not survive a round-trip to
    disk -- and raises a clear :class:`ValueError`.
    """
    if isinstance(previous_result_or_config, dict):
        config = previous_result_or_config
    else:
        config = getattr(previous_result_or_config, "attrs", {}).get("newsindex_config")

    if not config or "fn" not in config:
        raise ValueError(
            "rerun() needs a config: pass the DataFrame freshly returned by run_index()/"
            "run_index_with_baseline()/a quick-builder (its .attrs['newsindex_config'] "
            "carries it), or an explicit config dict with the same shape. A DataFrame "
            "reloaded from CSV or similar has no usable config -- attrs don't survive "
            "a round-trip to disk."
        )

    config = dict(config)
    fn = config.pop("fn")
    if verbose is not None:
        config["verbose"] = verbose

    if fn == "run_index":
        return run_index(**config)
    if fn == "run_index_with_baseline":
        return run_index_with_baseline(**config)
    raise ValueError(f"unknown config fn {fn!r}")
