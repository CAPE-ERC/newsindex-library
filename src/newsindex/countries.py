"""Global country reference: ISO 3166-1 codes and UN M49 regions.

A pure reference table, not consumed by any other function in this
library. It exists so a user building their own country groupings (for
:func:`newsindex.aggregate.aggregate_regions` or anything else) has a
standard, verified list of countries, codes, and regions to start from,
without compiling one by hand.

The 249 rows are every ISO 3166-1 entry (UN member states and
dependencies/territories alike), each carrying its UN M49 geographic
classification. That classification is three levels, and the third level
is only defined for some countries: ``continent`` (5 values: Africa,
Americas, Asia, Europe, Oceania), ``sub_region`` (17 values), and
``intermediate_region`` (7 values, set only for Africa and the Americas --
e.g. Nigeria's ``sub_region`` is "Sub-Saharan Africa" and its
``intermediate_region`` is "Western Africa"; a European country has no
``intermediate_region`` at all because its ``sub_region`` is already the
fine-grained level). ``continent``/``sub_region``/``intermediate_region``
are blank for the two ISO entries the UN M49 scheme does not classify
(Antarctica; Taiwan, which is not a UN member).

Names are ISO 3166-1's official short names, which do not always match
the names used elsewhere in this library (e.g. this table's "Congo,
Democratic Republic of the" is "Congo DRC" in
:data:`newsindex.aggregate.AFRICA_REGIONS` and the EPU dictionary) --
the same kind of mismatch :data:`newsindex.aggregate.DEFAULT_COUNTRY_ALIASES`
already bridges for GDP source files.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pandas as pd

_COLUMNS = ["name", "iso2", "iso3", "continent", "sub_region", "intermediate_region"]


def countries(path: str | Path | None = None) -> pd.DataFrame:
    """Load the country reference table.

    :param path: Optional override; defaults to the table bundled with
        this package.
    :return: A DataFrame with columns ``name``, ``iso2``, ``iso3``,
        ``continent``, ``sub_region``, ``intermediate_region``, one row
        per ISO 3166-1 entry.
    """
    if path is not None:
        return pd.read_csv(path, dtype=str, keep_default_na=False)[_COLUMNS]
    with resources.files("newsindex").joinpath("data", "countries.csv").open(
        "r", encoding="utf-8"
    ) as handle:
        return pd.read_csv(handle, dtype=str, keep_default_na=False)[_COLUMNS]
