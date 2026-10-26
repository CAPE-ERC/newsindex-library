"""Unicode-aware literal term matching and EPU classification.

Terms are matched as complete lexical units, not arbitrary substrings. This
prevents abbreviations such as EAC, EU, and OMC from matching inside longer
words, while whole phrases, apostrophes, hyphens, Arabic, and Ethiopic text
are matched correctly. Word boundaries are determined from the Unicode
general category of each character (letters, numbers, and combining marks
count as word characters), so the same logic works for scripts that do not
use ASCII word-boundary conventions.
"""

from __future__ import annotations

import functools
import math
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_DASHES = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


def is_missing(value: object) -> bool:
    """True for ``None`` and for float NaN (e.g. a missing value read from a
    pandas column), the two ways "no value" shows up in article data."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def normalize(value: object) -> str:
    """Return stable Unicode text used for both dictionary terms and articles.

    Applies Unicode NFKC normalization, folds apostrophe and dash variants to
    a single form, collapses whitespace, and case-folds the result.
    """
    text = "" if is_missing(value) else str(value)
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).translate(_DASHES)
    return re.sub(r"\s+", " ", text).casefold().strip()


def _is_word_character(character: str) -> bool:
    if not character:
        return False
    return unicodedata.category(character)[0] in {"L", "N", "M"}


@functools.lru_cache(maxsize=32768)
def _compiled_term(term: str) -> re.Pattern[str]:
    normalized = normalize(term)
    if not normalized:
        return re.compile(r"(?!x)x")
    pieces = re.split(r"([\s-]+)", normalized)
    body_parts: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if re.fullmatch(r"[\s-]+", piece):
            body_parts.append(r"(?:-|\s)+")
        else:
            body_parts.append(re.escape(piece))
    body = "".join(body_parts)
    left = r"(?<![^\W_])" if _is_word_character(normalized[0]) else ""
    right = r"(?![^\W_])" if _is_word_character(normalized[-1]) else ""
    return re.compile(left + body + right, flags=re.UNICODE)


def term_matches(normalized_text: str, term: str) -> bool:
    """Match one term against text already returned by :func:`normalize`."""
    return bool(_compiled_term(term).search(normalized_text))


def find_terms(text: object, terms: Iterable[str]) -> list[str]:
    """Return each distinct dictionary term found in ``text``, in input order."""
    normalized = normalize(text)
    seen: set[str] = set()
    matches: list[str] = []
    for value in terms:
        term = str(value).strip()
        key = normalize(term)
        if term and key not in seen and term_matches(normalized, term):
            seen.add(key)
            matches.append(term)
    return matches


def any_term_matches(text: object, terms: Iterable[str]) -> bool:
    """True if at least one of ``terms`` occurs in ``text``."""
    normalized = normalize(text)
    return any(term_matches(normalized, str(term)) for term in terms if str(term).strip())


def component_matches(
    title: object, body: object, components: Mapping[str, Sequence[str]]
) -> dict[str, bool]:
    """Return, for each named component, whether any of its terms occur."""
    full_text = f"{'' if is_missing(title) else title} {'' if is_missing(body) else body}"
    normalized = normalize(full_text)
    return {
        name: any(
            term_matches(normalized, str(term)) for term in terms if str(term).strip()
        )
        for name, terms in components.items()
    }


def is_epu(
    title: object,
    body: object,
    economic_terms: Sequence[str],
    policy_terms: Sequence[str],
    uncertainty_terms: Sequence[str],
) -> bool:
    """True if the article contains at least one Economic, Policy, and
    Uncertainty term (Baker, Bloom and Davis, 2016 classification rule)."""
    components = {
        "economic": economic_terms,
        "policy": policy_terms,
        "uncertainty": uncertainty_terms,
    }
    flags = component_matches(title, body, components)
    return flags["economic"] and flags["policy"] and flags["uncertainty"]
