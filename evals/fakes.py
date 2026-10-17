"""Fakes for driving the agent without the real Anthropic or Tavily APIs.

The fakes sit below the reliability layer: FakeAnthropic replaces the SDK
client inside the real LLMClient, and the fake fetch/search functions replace
the network calls inside the real ToolExecutor. Retries, timeouts, fallbacks
and validation all run for real.
"""

import asyncio
import copy
from collections.abc import Callable
from typing import Any

import anthropic
import httpx

from agent.llm import LLMClient
from agent.orchestrator import ResearchAgent
from agent.reliability import RetryPolicy
from agent.tools import SearchHit, ToolError, ToolExecutor


class FakeBlock:
    """Mimics an Anthropic content block (text or tool_use)."""

    def __init__(self, type_, text=None, name=None, input=None, id=None):
        self.type = type_
        if text is not None:
            self.text = text
        self.name = name
        self.input = input
        self.id = id


class FakeUsage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


class FakeStopDetails:
    def __init__(self, category):
        self.type = "refusal"
        self.category = category
        self.explanation = None


class FakeResponse:
    """Mimics an Anthropic Message response."""

    def __init__(self, stop_reason, content, usage=None, model=None, stop_details=None):
        self.stop_reason = stop_reason
        self.content = content
        self.usage = usage or FakeUsage()
        self.model = model
        self.stop_details = stop_details


# Token counts are scripted to the rough shape of a real run: the input grows
# each turn as tool results accumulate, the final turn writes the brief.
TOOL_TURN_USAGE = (2_400, 160)
FINAL_TURN_USAGE = (6_800, 1_100)


def text_response(text: str, usage=FINAL_TURN_USAGE) -> FakeResponse:
    return FakeResponse(
        stop_reason="end_turn", content=[FakeBlock("text", text=text)], usage=FakeUsage(*usage)
    )


def tool_use_response(tool_calls: list[tuple[str, dict]], usage=TOOL_TURN_USAGE) -> FakeResponse:
    """tool_calls: list of (tool_name, tool_input) pairs."""
    blocks = []
    for i, (name, tool_input) in enumerate(tool_calls):
        blocks.append(FakeBlock("tool_use", name=name, input=tool_input, id=f"tool_{name}_{i}"))
    return FakeResponse(stop_reason="tool_use", content=blocks, usage=FakeUsage(*usage))


def refusal_response(category="cyber") -> FakeResponse:
    return FakeResponse(
        stop_reason="refusal",
        content=[],
        usage=FakeUsage(900, 0),
        stop_details=FakeStopDetails(category),
    )


def api_status_error(status: int, message: str = "error") -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return anthropic.APIStatusError(message, response=response, body=None)


def overloaded_error() -> anthropic.APIStatusError:
    return api_status_error(529, "Overloaded")


HANG = object()
"""Script item: the call never returns, so only a timeout can end it."""


class FakeMessages:
    """Replays a script of responses. Each item is a FakeResponse, an
    exception to raise, HANG, or a callable(request_kwargs) returning either.
    When the script runs out the last item repeats, so max_iterations
    behaviour can be exercised."""

    def __init__(self, script: list[Any]):
        self._script = list(script)
        self._index = 0
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        # Snapshot the history: the orchestrator keeps appending to the list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        if callable(item):
            item = item(kwargs)
        if item is HANG:
            await asyncio.sleep(3600)
        if isinstance(item, BaseException):
            raise item
        # Fresh copy each call: the orchestrator appends response.content to
        # the message history, and a reused object would alias across turns.
        response = copy.copy(item)
        response.content = list(item.content)
        response.model = item.model or kwargs["model"]
        return response


class FakeAnthropic:
    def __init__(self, script: list[Any]):
        self.messages = FakeMessages(script)


class Seq:
    """A canned tool outcome per call; the last one repeats."""

    def __init__(self, *outcomes):
        self.outcomes = outcomes
        self.calls = 0

    def next(self):
        item = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        return item


async def no_sleep(_seconds: float) -> None:
    return None


def _resolve(value, arg):
    if isinstance(value, Seq):
        value = value.next()
    if callable(value):
        value = value(arg)
    if value is HANG:
        return HANG
    if isinstance(value, BaseException):
        raise value
    return value


def make_tool_executor(canned: dict, timeout: float = 5.0) -> ToolExecutor:
    """canned: {"fetch_url": str | ToolError | Seq | callable(url),
    "search_web": list[SearchHit] | ToolError | Seq | callable(query)}"""

    async def fetch(url: str) -> str:
        value = _resolve(canned.get("fetch_url", ToolError("no canned page", retryable=False)), url)
        if value is HANG:
            await asyncio.sleep(3600)
        return value

    async def search(query: str, max_results: int) -> list[SearchHit]:
        value = _resolve(canned.get("search_web", []), query)
        if value is HANG:
            await asyncio.sleep(3600)
        return value[:max_results]

    return ToolExecutor(
        fetch=fetch,
        search=search,
        retry_policy=RetryPolicy(max_attempts=3, jitter=False),
        timeout=timeout,
        sleep=no_sleep,
    )


def make_agent(
    script: list[Any],
    canned_tools: dict,
    *,
    model: str = "claude-opus-5-5",
    fallback_model: str = "claude-sonnet-5-5",
    llm_timeout: float = 5.0,
    tool_timeout: float = 5.0,
    **agent_kwargs,
) -> tuple[ResearchAgent, FakeAnthropic]:
    fake = FakeAnthropic(script)
    llm = LLMClient(
        fake,
        model=model,
        fallback_model=fallback_model,
        retry_policy=RetryPolicy(max_attempts=3, jitter=False),
        timeout=llm_timeout,
        sleep=no_sleep,
    )
    return ResearchAgent(llm, make_tool_executor(canned_tools, tool_timeout), **agent_kwargs), fake


def by_model(responses: dict[str, Callable[[], Any]]):
    """Script item that answers differently per requested model."""

    def _item(kwargs):
        return responses[kwargs["model"]]()

    return _item


# ─── Canned research corpus ──────────────────────────────────────────────────
# A fictional company. Quotes in the briefs below are copied from this text,
# which is what the groundedness checks verify.

JOB_URL = "https://example.com/job/123"
JOB_PAGE = (
    "Senior Backend Engineer at Acme Corp. Remote (US time zones). "
    "We build with Python, FastAPI and PostgreSQL. "
    "You have 5+ years of Python and experience with distributed systems. "
    "You will own API design for our payments platform."
)
ABOUT_URL = "https://acme.example/about"
NEWS_URL = "https://news.example/acme-series-b"
ABOUT_HIT = SearchHit(
    "Acme Corp Overview",
    ABOUT_URL,
    "Acme Corp is a Series B fintech founded in 2019, building payment infrastructure "
    "for small businesses. The team of 30 people works async-first with no mandatory meetings.",
)
NEWS_HIT = SearchHit(
    "Acme Corp raises Series B",
    NEWS_URL,
    "Acme Corp raised a $40M Series B in 2025 to expand its payments platform into Europe.",
)
SEARCH_HITS = [ABOUT_HIT, NEWS_HIT]

VALID_BRIEF = {
    "role": "Senior Backend Engineer",
    "company": "Acme Corp",
    "location": "Remote (US time zones)",
    "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
    "key_requirements": ["5+ years Python", "Distributed systems experience", "API design"],
    "company_summary": "Acme Corp is a Series B fintech founded in 2019, building payment infrastructure for small businesses.",
    "culture_signals": ["Async-first, no mandatory meetings", "Small team (30 people)"],
    "talking_points": ["Highlight FastAPI experience", "Mention distributed systems project"],
    "red_flags": [],
    "sources": [JOB_URL, ABOUT_URL, NEWS_URL],
    "claims": [
        {
            "statement": "The role is remote within US time zones.",
            "source": JOB_URL,
            "quote": "Remote (US time zones).",
        },
        {
            "statement": "Acme Corp is a Series B fintech founded in 2019.",
            "source": ABOUT_URL,
            "quote": "Acme Corp is a Series B fintech founded in 2019",
        },
        {
            "statement": "The team of 30 people works async-first with no mandatory meetings.",
            "source": ABOUT_URL,
            "quote": "The team of 30 people works async-first with no mandatory meetings.",
        },
        {
            "statement": "Acme Corp raised a $40M Series B in 2025.",
            "source": NEWS_URL,
            "quote": "Acme Corp raised a $40M Series B in 2025",
        },
    ],
}

CANNED_TOOLS = {"fetch_url": JOB_PAGE, "search_web": SEARCH_HITS}


def research_turns() -> list[FakeResponse]:
    """The standard opening: read the posting, then search for the company."""
    return [
        tool_use_response([("fetch_url", {"url": JOB_URL})]),
        tool_use_response([("search_web", {"query": "Acme Corp company overview funding 2025"})]),
    ]
