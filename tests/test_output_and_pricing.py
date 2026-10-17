import json

import pytest

from agent.output import BriefValidationError, extract_json, normalize_url, parse_brief
from agent.pricing import call_cost
from agent.usage import RunUsage
from evals.fakes import ABOUT_URL, JOB_URL, NEWS_URL, VALID_BRIEF, FakeUsage

RETRIEVED = {JOB_URL, ABOUT_URL, NEWS_URL}


def brief(**changes) -> dict:
    data = json.loads(json.dumps(VALID_BRIEF))
    data.update(changes)
    return data


# ─── extract_json / parse_brief ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(VALID_BRIEF),
        "```json\n" + json.dumps(VALID_BRIEF) + "\n```",
        "Here you go:\n" + json.dumps(VALID_BRIEF) + "\nThanks!",
    ],
)
def test_extract_json_handles_clean_fenced_and_wrapped(text):
    assert json.loads(extract_json(text))["company"] == "Acme Corp"


def test_parse_brief_accepts_valid_brief():
    parsed = parse_brief(json.dumps(VALID_BRIEF), retrieved_urls=RETRIEVED)
    assert parsed.company == "Acme Corp"
    assert len(parsed.claims) == 4


def test_parse_brief_reports_missing_fields_by_name():
    data = brief()
    del data["sources"]
    with pytest.raises(BriefValidationError) as exc:
        parse_brief(json.dumps(data))
    assert any(p.startswith("sources") for p in exc.value.problems)


def test_parse_brief_rejects_prose():
    with pytest.raises(BriefValidationError, match="not a JSON object"):
        parse_brief("I could not find enough information.")


@pytest.mark.parametrize(
    "changes,problem",
    [
        ({"role": "  "}, "role"),
        ({"sources": ["not a url"]}, "sources"),
        ({"claims": []}, "claims"),
        ({"sources": [JOB_URL]}, "not listed in 'sources'"),
    ],
)
def test_parse_brief_validators(changes, problem):
    with pytest.raises(BriefValidationError) as exc:
        parse_brief(json.dumps(brief(**changes)))
    assert problem in str(exc.value)


def test_parse_brief_rejects_claims_citing_unretrieved_urls():
    with pytest.raises(BriefValidationError, match="no tool call in this run retrieved"):
        parse_brief(json.dumps(VALID_BRIEF), retrieved_urls={JOB_URL})


def test_url_matching_ignores_trailing_slash_fragment_and_host_case():
    assert normalize_url("HTTPS://Acme.Example/about/#team") == normalize_url(ABOUT_URL)
    parse_brief(
        json.dumps(VALID_BRIEF),
        retrieved_urls={JOB_URL + "/", "https://ACME.example/about#top", NEWS_URL},
    )


# ─── pricing / usage ─────────────────────────────────────────────────────────


def test_call_cost_standard_and_cache_rates():
    # Opus 5.5: $4 in, $20 out, $0.20 cache read, $5 cache write (1.25x input)
    cost = call_cost(
        "claude-opus-5-5",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost == pytest.approx(4.00 + 2.00 + 0.20 + 5.00)


def test_call_cost_long_context_tier():
    short = call_cost("claude-haiku-5-5", input_tokens=50_000, output_tokens=1_000)
    long = call_cost("claude-haiku-5-5", input_tokens=150_000, output_tokens=1_000)
    assert short == pytest.approx(50_000 * 0.10 / 1e6 + 1_000 * 0.50 / 1e6)
    assert long == pytest.approx(150_000 * 0.50 / 1e6 + 1_000 * 2.50 / 1e6)


def test_call_cost_unknown_model_is_none():
    assert call_cost("some-future-model", input_tokens=1, output_tokens=1) is None


def test_run_usage_aggregates_per_model():
    usage = RunUsage()
    usage.record("claude-opus-5-5", FakeUsage(1_000, 100, cache_read=500))
    usage.record("claude-opus-5-5", FakeUsage(2_000, 200))
    usage.record("claude-sonnet-5-5", FakeUsage(1_000, 100))
    usage.record("mystery-model", FakeUsage(10, 10))
    d = usage.to_dict()
    assert d["by_model"]["claude-opus-5-5"]["calls"] == 2
    assert d["by_model"]["claude-opus-5-5"]["cache_read_tokens"] == 500
    assert d["input_tokens"] == 1_000 + 500 + 2_000 + 1_000 + 10
    assert d["output_tokens"] == 410
    assert d["cost_known"] is False
    opus = (3_000 * 4 + 300 * 20 + 500 * 0.20) / 1e6
    sonnet = (1_000 * 2 + 100 * 10) / 1e6
    assert d["cost_usd"] == pytest.approx(opus + sonnet, abs=1e-6)


def test_run_usage_tolerates_missing_usage():
    usage = RunUsage()
    call = usage.record("claude-opus-5-5", None)
    assert call.cost_usd == 0
