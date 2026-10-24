import pandas as pd

from newsindex import dedup


def test_canonical_url_normalizes_case_tracking_params_and_trailing_slash():
    a = dedup.canonical_url("HTTPS://Example.COM/News/Story/?utm_source=x&id=1")
    b = dedup.canonical_url("https://example.com/News/Story?id=1")
    assert a == b


def test_canonical_url_handles_missing_url():
    assert dedup.canonical_url(None) == ""
    assert dedup.canonical_url("") == ""
    assert dedup.canonical_url("   ") == ""


def test_dedupe_collapses_shared_canonical_url_within_group():
    articles = pd.DataFrame(
        {
            "country": ["Nigeria", "Nigeria", "Nigeria"],
            "year_month": ["2026-03", "2026-03", "2026-03"],
            "url": [
                "https://example.com/a?utm_source=twitter",
                "https://example.com/a",
                "https://example.com/b",
            ],
            "title": ["A", "A (reshared)", "B"],
            "text": ["text a", "text a", "text b"],
        }
    )
    result = dedup.dedupe(articles)
    assert len(result) == 2
    assert set(result["title"]) == {"A", "B"}


def test_dedupe_falls_back_to_title_and_text_hash_when_url_missing():
    articles = pd.DataFrame(
        {
            "country": ["Kenya", "Kenya"],
            "year_month": ["2026-01", "2026-01"],
            "url": [None, ""],
            "title": ["Same story", "Same story"],
            "text": ["identical body", "identical body"],
        }
    )
    result = dedup.dedupe(articles)
    assert len(result) == 1


def test_dedupe_does_not_merge_across_month_boundary():
    articles = pd.DataFrame(
        {
            "country": ["Kenya", "Kenya"],
            "year_month": ["2026-01", "2026-02"],
            "url": ["https://example.com/a", "https://example.com/a"],
            "title": ["A", "A"],
            "text": ["text", "text"],
        }
    )
    result = dedup.dedupe(articles)
    assert len(result) == 2
