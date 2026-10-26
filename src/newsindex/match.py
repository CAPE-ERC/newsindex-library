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


def _term_body(normalized: str) -> str:
    """The escaped, separator-flexible middle of a term's pattern: its
    words joined by ``(?:-|\\s)+`` so a dash and a space are interchangeable
    between them, same as :func:`normalize` already folds dash variants and
    collapses whitespace within a single string. No boundary assertions."""
    pieces = re.split(r"([\s-]+)", normalized)
    body_parts: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if re.fullmatch(r"[\s-]+", piece):
            body_parts.append(r"(?:-|\s)+")
        else:
            body_parts.append(re.escape(piece))
    return "".join(body_parts)


@functools.lru_cache(maxsize=32768)
def _compiled_term(term: str) -> re.Pattern[str]:
    normalized = normalize(term)
    if not normalized:
        return re.compile(r"(?!x)x")
    body = _term_body(normalized)
    left = r"(?<![^\W_])" if _is_word_character(normalized[0]) else ""
    right = r"(?![^\W_])" if _is_word_character(normalized[-1]) else ""
    return re.compile(left + body + right, flags=re.UNICODE)


@functools.lru_cache(maxsize=32768)
def _loose_term_body(term: str) -> str:
    normalized = normalize(term)
    if not normalized:
        return r"(?!x)x"
    return _term_body(normalized)


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


@functools.lru_cache(maxsize=4096)
def _combined_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    bodies: list[str] = []
    seen: set[str] = set()
    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        key = normalize(term)
        if key in seen:
            continue
        compiled = _compiled_term(term)
        if compiled.pattern == r"(?!x)x":
            continue
        seen.add(key)
        bodies.append(compiled.pattern)
    if not bodies:
        return re.compile(r"(?!x)x")
    return re.compile("|".join(f"(?:{body})" for body in bodies), flags=re.UNICODE)


def component_pattern(terms: Sequence[str]) -> re.Pattern[str]:
    """One compiled regex matching normalized text that contains any of ``terms``.

    Equivalent to ``any(term_matches(text, term) for term in terms)`` but as
    a single precompiled alternation, so a whole column of articles can be
    tested against a component in one vectorized pass (e.g. pandas
    ``Series.str.contains``) instead of one Python-level call per term per
    row. Memoized per distinct ``terms`` tuple, since the same dictionary
    component is reused across every row and batch of a run.

    This still carries each term's word-boundary assertions, which cost
    real regex-engine time per character scanned. When most rows won't
    contain any of ``terms`` at all (the common case classifying real
    articles), check :func:`loose_component_pattern` first and only run
    this on the rows it flags -- see its docstring.
    """
    return _combined_pattern(tuple(terms))


@functools.lru_cache(maxsize=4096)
def _combined_loose_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    bodies: list[str] = []
    seen: set[str] = set()
    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        key = normalize(term)
        if key in seen:
            continue
        body = _loose_term_body(term)
        if body == r"(?!x)x":
            continue
        seen.add(key)
        bodies.append(body)
    if not bodies:
        return re.compile(r"(?!x)x")
    return re.compile("|".join(f"(?:{body})" for body in bodies), flags=re.UNICODE)


def loose_component_pattern(terms: Sequence[str]) -> re.Pattern[str]:
    """A cheap, over-inclusive version of :func:`component_pattern`.

    Same terms, same dash/whitespace flexibility between words, but without
    the word-boundary lookaround assertions -- so this can only produce
    false positives relative to :func:`component_pattern`, never false
    negatives (dropping a check can't make a pattern match less). Matching
    it is substantially cheaper per character scanned, since there's no
    lookaround to evaluate at every position.

    Use it as a prefilter: test a whole column with this pattern first
    (e.g. pandas ``Series.str.contains``), then run
    :func:`component_pattern` only on the rows it flagged. Most articles
    don't contain any term from a given component at all, so this avoids
    ever running the expensive pattern over the great majority of rows.
    """
    return _combined_loose_pattern(tuple(terms))


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
