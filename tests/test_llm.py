import anthropic
import httpx
import pytest

from agent.llm import REFUSAL_FALLBACK_BETA, LLMClient, is_retryable_llm_error
from agent.reliability import RetryPolicy
from agent.usage import RunUsage
from evals.fakes import FakeAnthropic, api_status_error, overloaded_error, text_response
from tests.conftest import by_name


async def no_sleep(_):
    return None


def client(script, **kwargs) -> tuple[LLMClient, FakeAnthropic]:
    fake = FakeAnthropic(script)
    llm = LLMClient(
        fake,
        model=kwargs.pop("model", "claude-opus-5-5"),
        fallback_model=kwargs.pop("fallback_model", "claude-sonnet-5-5"),
        retry_policy=RetryPolicy(max_attempts=3, jitter=False),
        timeout=1.0,
        sleep=no_sleep,
        **kwargs,
    )
    return llm, fake


async def create(llm: LLMClient, usage: RunUsage | None = None):
    return await llm.create(
        system="sys", tools=[], messages=[{"role": "user", "content": "hi"}], usage=usage
    )


@pytest.mark.parametrize(
    "exc,retryable",
    [
        (api_status_error(400), False),
        (api_status_error(401), False),
        (api_status_error(404), False),
        (api_status_error(408), True),
        (api_status_error(429), True),
        (api_status_error(500), True),
        (overloaded_error(), True),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), True),
        (TimeoutError(), True),
        (ValueError(), False),
    ],
)
def test_retryable_classification(exc, retryable):
    assert is_retryable_llm_error(exc) is retryable


def test_request_params_per_model():
    llm, _ = client([], effort="high")
    opus = llm.request_params("claude-opus-5-5")
    assert opus["output_config"] == {"effort": "high"}
    assert opus["extra_headers"] == {"anthropic-beta": REFUSAL_FALLBACK_BETA}
    assert opus["extra_body"] == {"fallbacks": "default"}

    haiku = llm.request_params("claude-haiku-5-5")
    assert haiku["output_config"] == {"effort": "high"}
    assert "extra_body" not in haiku  # no server-side refusal fallback on Haiku

    old = llm.request_params("claude-haiku-4-5")
    assert "output_config" not in old  # effort is rejected there

    no_fallback, _ = client([], refusal_fallback=False)
    assert "extra_body" not in no_fallback.request_params("claude-opus-5-5")


async def test_retries_overloaded_then_succeeds(spans):
    llm, fake = client([overloaded_error(), text_response("done")])
    call = await create(llm)
    assert call.attempts == 2
    assert call.model == "claude-opus-5-5"
    assert not call.fell_back
    assert len(fake.messages.requests) == 2
    (span,) = by_name(spans, "chat claude-opus-5-5")
    assert span.attributes["llm.attempts"] == 2
    assert [e.name for e in span.events] == ["retry"]


async def test_falls_back_and_stays_on_fallback_model(spans):
    def respond(kwargs):
        if kwargs["model"] == "claude-opus-5-5":
            return overloaded_error()
        return text_response("from fallback")

    llm, fake = client([respond])
    first = await create(llm)
    assert first.fell_back and first.model == "claude-sonnet-5-5"
    assert first.attempts == 4  # 3 on the primary, 1 on the fallback

    second = await create(llm)
    assert not second.fell_back and second.model == "claude-sonnet-5-5"
    assert [r["model"] for r in fake.messages.requests][-1] == "claude-sonnet-5-5"
    assert len(fake.messages.requests) == 5  # no retries wasted on the primary

    failed_primary = by_name(spans, "chat claude-opus-5-5")
    assert failed_primary[0].status.is_ok is False


async def test_non_retryable_error_does_not_fall_back():
    llm, fake = client([api_status_error(400, "bad request")])
    with pytest.raises(anthropic.APIStatusError):
        await create(llm)
    assert len(fake.messages.requests) == 1
    assert llm.active_model == "claude-opus-5-5"


async def test_no_fallback_configured_reraises():
    llm, _ = client([overloaded_error()], fallback_model="")
    with pytest.raises(anthropic.APIStatusError):
        await create(llm)


async def test_records_usage_and_cost_on_span(spans):
    llm, _ = client([text_response("done", usage=(1_000, 500))])
    usage = RunUsage()
    await create(llm, usage)
    (span,) = by_name(spans, "chat claude-opus-5-5")
    assert span.attributes["gen_ai.usage.input_tokens"] == 1_000
    assert span.attributes["gen_ai.usage.output_tokens"] == 500
    assert span.attributes["gen_ai.request.model"] == "claude-opus-5-5"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("end_turn",)
    # 1000 * $4/M + 500 * $20/M
    assert span.attributes["llm.cost_usd"] == pytest.approx(0.014)
    assert usage.by_model["claude-opus-5-5"].calls == 1
