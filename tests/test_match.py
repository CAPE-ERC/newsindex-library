from newsindex import match


def test_normalize_folds_case_apostrophes_and_dashes():
    assert match.normalize("Government’s  POLICY—shift") == "government's policy-shift"


def test_normalize_handles_none_and_non_strings():
    assert match.normalize(None) == ""
    assert match.normalize(42) == "42"


def test_normalize_handles_float_nan_like_none():
    # float NaN is how a missing pandas cell can surface (e.g. a title or
    # body column with some blank rows); it must not be stringified into
    # the literal text "nan".
    assert match.normalize(float("nan")) == ""


def test_is_epu_treats_missing_title_or_body_as_empty_not_literal_nan():
    economic, policy, uncertainty = ["economy"], ["government"], ["uncertainty"]
    assert not match.is_epu(float("nan"), "", economic, policy, uncertainty)
    assert match.is_epu(
        "Government economy uncertainty", float("nan"), economic, policy, uncertainty
    )


def test_short_abbreviations_do_not_match_inside_words():
    assert not match.any_term_matches("Each country acts", ["EAC"])
    assert not match.any_term_matches("The newcomer arrived", ["OMC"])
    assert not match.any_term_matches("A neural network", ["EU"])


def test_complete_abbreviations_and_phrases_match():
    assert match.any_term_matches("An EAC trade agreement", ["EAC"])
    assert match.any_term_matches("free-trade agreement signed", ["free trade agreement"])
    assert match.any_term_matches("free trade agreement signed", ["free-trade agreement"])


def test_arabic_and_portuguese_terms_match_with_word_boundaries():
    assert match.any_term_matches(
        "أزمة اقتصادية مع تغييرات "
        "في السياسة التجارية",
        ["أزمة اقتصادية"],
    )
    assert match.any_term_matches("incerteza economica", ["incerteza"])
    assert not match.any_term_matches("incertezabilidade", ["incerteza"])


def test_find_terms_returns_distinct_matches_in_order_no_duplicates():
    terms = ["economy", "inflation", "policy", "economy"]
    found = match.find_terms("The economy faces inflation amid new policy and economy news", terms)
    assert found == ["economy", "inflation", "policy"]


def test_is_epu_requires_all_three_components():
    economic, policy, uncertainty = ["economy", "inflation"], ["government", "fiscal policy"], ["uncertainty", "volatile"]
    assert match.is_epu(
        "Government policy creates uncertainty", "in the economy", economic, policy, uncertainty
    )
    assert not match.is_epu("The economy is growing", "", economic, policy, uncertainty)
    assert not match.is_epu(
        "Uncertainty remains after the football match", "", economic, policy, uncertainty
    )
