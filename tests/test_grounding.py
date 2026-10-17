from evals.fakes import ABOUT_HIT, ABOUT_URL, JOB_PAGE, JOB_URL, NEWS_HIT, NEWS_URL, VALID_BRIEF
from evals.grounding import grounding_report, score_claim

CORPUS = {
    JOB_URL: JOB_PAGE,
    ABOUT_URL: f"{ABOUT_HIT.title}\n{ABOUT_HIT.content}",
    NEWS_URL: f"{NEWS_HIT.title}\n{NEWS_HIT.content}",
}


def claim(statement, source, quote):
    return {"statement": statement, "source": source, "quote": quote}


def test_valid_brief_is_fully_grounded():
    report = grounding_report(VALID_BRIEF, CORPUS)
    assert report["claims"] == 4
    assert report["citation_accuracy"] == 1.0
    assert report["groundedness"] == 1.0
    assert report["source_precision"] == 1.0
    assert report["unsupported"] == []


def test_unretrieved_source_fails_citation_and_support():
    s = score_claim(
        claim("Example Co is profitable.", "https://elsewhere.example", "profitable"), CORPUS
    )
    assert not s.cites_retrieved_source
    assert not s.supported
    assert s.reason == "source not retrieved"


def test_fabricated_quote_is_unsupported():
    s = score_claim(
        claim(
            "Example Co was founded in 2019.",
            ABOUT_URL,
            "Example Co was founded in 2019 by two engineers",
        ),
        CORPUS,
    )
    assert s.cites_retrieved_source
    assert not s.quote_found
    assert not s.supported


def test_changed_number_is_unsupported_even_with_a_real_quote():
    s = score_claim(claim("Example Co raised $50M.", NEWS_URL, "raised a $40M Series B"), CORPUS)
    assert s.quote_found
    assert not s.numbers_supported
    assert "50" in s.reason


def test_unrelated_statement_with_real_quote_has_low_overlap():
    s = score_claim(
        claim(
            "Leadership plans layoffs and offshoring next quarter.", ABOUT_URL, "founded in 2019"
        ),
        CORPUS,
    )
    assert s.quote_found
    assert s.content_overlap < 0.5
    assert not s.supported


def test_matching_ignores_case_whitespace_curly_quotes_and_url_form():
    corpus = {"https://site.example/page": "We’re   hiring a\nStaff Engineer — remote."}
    s = score_claim(
        claim(
            "They're hiring a staff engineer, remote.",
            "https://SITE.example/page/",
            "we're hiring a staff engineer - remote",
        ),
        corpus,
    )
    assert s.supported, s.reason


def test_report_on_brief_without_claims():
    report = grounding_report({"claims": [], "sources": []}, CORPUS)
    assert report["groundedness"] is None
    assert report["citation_accuracy"] is None
