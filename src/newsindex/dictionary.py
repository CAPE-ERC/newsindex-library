"""Load and validate country keyword dictionaries for EPU classification.

The on-disk format is the EPTU schema used across CAPE's EPU, ETU, and TPU
indices: one JSON document with a ``countries`` map, each entry carrying a
language list and Economic / Policy / Trade / Uncertainty term lists. A term
belongs to exactly one component; :func:`validate` checks that no literal
term, once normalized, is shared across the Economic, Policy, and
Uncertainty components used by the EPU index.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .match import normalize

_FIELD_MAP = {
    "economic": "ECONOMY_TERMS",
    "policy": "POLICY_TERMS",
    "uncertainty": "UNCERTAINTY_TERMS",
    "trade": "TRADE_TERMS",
}
_EPU_COMPONENTS = ("economic", "policy", "uncertainty")


@dataclass(frozen=True)
class CountryDictionary:
    """One country's keyword lists."""

    country: str
    languages: tuple[str, ...]
    economic: tuple[str, ...]
    policy: tuple[str, ...]
    uncertainty: tuple[str, ...]
    trade: tuple[str, ...] = ()


@dataclass(frozen=True)
class Dictionary:
    """A loaded, country-keyed keyword dictionary."""

    countries: dict[str, CountryDictionary]
    schema: str = ""
    source: Path | None = None
    meta: dict = field(default_factory=dict)

    def __getitem__(self, country: str) -> CountryDictionary:
        return self.countries[country]

    def __contains__(self, country: str) -> bool:
        return country in self.countries

    def __iter__(self):
        return iter(self.countries)

    def __len__(self) -> int:
        return len(self.countries)


def load(path: str | Path) -> Dictionary:
    """Load a country keyword dictionary from an EPTU-schema JSON file."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("countries", raw)
    countries: dict[str, CountryDictionary] = {}
    for country, payload in rows.items():
        countries[country] = CountryDictionary(
            country=country,
            languages=tuple(payload.get("languages", [])),
            economic=tuple(payload.get(_FIELD_MAP["economic"], [])),
            policy=tuple(payload.get(_FIELD_MAP["policy"], [])),
            uncertainty=tuple(payload.get(_FIELD_MAP["uncertainty"], [])),
            trade=tuple(payload.get(_FIELD_MAP["trade"], [])),
        )
    meta = raw.get("_meta", {}) if isinstance(raw, dict) else {}
    return Dictionary(countries=countries, schema=meta.get("schema", ""), source=path, meta=meta)


def validate(dictionary: Dictionary) -> list[dict]:
    """Audit a dictionary for empty EPU components and cross-component overlap.

    Returns a list of issue rows (empty if the dictionary is clean), each with
    ``country``, ``issue``, and supporting detail. A term that normalizes to
    the same text in two of {economic, policy, uncertainty} violates the rule
    that one term may satisfy only one index component.
    """
    issues: list[dict] = []
    for country, entry in dictionary.countries.items():
        component_terms = {
            name: getattr(entry, name) for name in _EPU_COMPONENTS
        }
        for name, terms in component_terms.items():
            if not terms:
                issues.append({"country": country, "issue": "empty_component", "component": name})
        normalized = {
            name: {normalize(term) for term in terms if str(term).strip()}
            for name, terms in component_terms.items()
        }
        names = list(normalized)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = normalized[left] & normalized[right]
                if overlap:
                    issues.append(
                        {
                            "country": country,
                            "issue": "cross_component_overlap",
                            "components": f"{left}/{right}",
                            "terms": sorted(overlap),
                        }
                    )
    return issues


def fingerprint(dictionary: Dictionary) -> str:
    """Return a stable hash identifying the exact dictionary contents.

    Two dictionaries with the same country, language, and term content hash
    identically regardless of key order; any change to a term list, a
    language list, or the set of countries changes the fingerprint. Baselines
    record this value so a later, differently worded dictionary cannot be
    silently reused to score frozen history.
    """
    canonical = {
        country: {
            "languages": sorted(entry.languages),
            "economic": sorted(normalize(t) for t in entry.economic),
            "policy": sorted(normalize(t) for t in entry.policy),
            "uncertainty": sorted(normalize(t) for t in entry.uncertainty),
            "trade": sorted(normalize(t) for t in entry.trade),
        }
        for country, entry in sorted(dictionary.countries.items())
    }
    blob = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
