import json
from pathlib import Path

import pytest

from newsindex import dictionary

DATA_DICTIONARY = Path(__file__).resolve().parents[1] / "data" / "eptu_country_keywords.json"


@pytest.fixture(scope="module")
def production_dictionary() -> dictionary.Dictionary:
    return dictionary.load(DATA_DICTIONARY)


def test_load_production_dictionary_covers_all_54_countries(production_dictionary):
    assert len(production_dictionary) == 54
    nigeria = production_dictionary["Nigeria"]
    assert nigeria.languages == ("en",)
    assert "economy" in nigeria.economic
    assert len(nigeria.policy) > 0
    assert len(nigeria.uncertainty) > 0


def test_production_dictionary_has_no_empty_or_overlapping_epu_components(production_dictionary):
    issues = dictionary.validate(production_dictionary)
    assert issues == []


def test_fingerprint_is_stable_and_order_independent(tmp_path):
    payload = {
        "countries": {
            "B": {"languages": ["en"], "ECONOMY_TERMS": ["gdp"], "POLICY_TERMS": ["tax"], "UNCERTAINTY_TERMS": ["risk"]},
            "A": {"languages": ["en"], "ECONOMY_TERMS": ["gdp"], "POLICY_TERMS": ["tax"], "UNCERTAINTY_TERMS": ["risk"]},
        }
    }
    reordered = {
        "countries": {
            "A": payload["countries"]["A"],
            "B": payload["countries"]["B"],
        }
    }
    path1 = tmp_path / "one.json"
    path2 = tmp_path / "two.json"
    path1.write_text(json.dumps(payload), encoding="utf-8")
    path2.write_text(json.dumps(reordered), encoding="utf-8")

    assert dictionary.fingerprint(dictionary.load(path1)) == dictionary.fingerprint(dictionary.load(path2))


def test_fingerprint_changes_when_a_term_changes(tmp_path):
    base = {
        "countries": {
            "A": {"languages": ["en"], "ECONOMY_TERMS": ["gdp"], "POLICY_TERMS": ["tax"], "UNCERTAINTY_TERMS": ["risk"]}
        }
    }
    changed = json.loads(json.dumps(base))
    changed["countries"]["A"]["ECONOMY_TERMS"] = ["gdp", "inflation"]

    path1 = tmp_path / "base.json"
    path2 = tmp_path / "changed.json"
    path1.write_text(json.dumps(base), encoding="utf-8")
    path2.write_text(json.dumps(changed), encoding="utf-8")

    assert dictionary.fingerprint(dictionary.load(path1)) != dictionary.fingerprint(dictionary.load(path2))


def test_validate_flags_empty_component_and_cross_component_overlap(tmp_path):
    payload = {
        "countries": {
            "A": {
                "languages": ["en"],
                "ECONOMY_TERMS": ["gdp", "risk"],
                "POLICY_TERMS": ["tax"],
                "UNCERTAINTY_TERMS": ["risk"],
            },
            "B": {
                "languages": ["en"],
                "ECONOMY_TERMS": [],
                "POLICY_TERMS": ["tax"],
                "UNCERTAINTY_TERMS": ["risk"],
            },
        }
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    issues = dictionary.validate(dictionary.load(path))
    issue_types = {(issue["country"], issue["issue"]) for issue in issues}
    assert ("A", "cross_component_overlap") in issue_types
    assert ("B", "empty_component") in issue_types
