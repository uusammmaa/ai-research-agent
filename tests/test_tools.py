import asyncio

import httpx
import pytest
from tavily import InvalidAPIKeyError

from agent import tools
from agent.reliability import Deadline, RetryPolicy, RunTimeoutError
from agent.tools import (
    MAX_PAGE_CHARS,
    SearchHit,
    ToolError,
    ToolExecutor,
    fetch_page,
    format_search_hits,
    tavily_search,
    validate_public_url,
)
from tests.conftest import by_name


async def no_sleep(_):
    return None


def executor(fetch=None, search=None, **kwargs) -> ToolExecutor:
    async def default_fetch(url):
        return f"page {url}"

    async def default_search(query, max_results):
        return [SearchHit("Result", "https://result.example/1", f"about {query}")]

    return ToolExecutor(
        fetch=fetch or default_fetch,
        search=search or default_search,
        retry_policy=RetryPolicy(max_attempts=3, jitter=False),
        timeout=kwargs.pop("timeout", 1.0),
        sleep=no_sleep,
        **kwargs,
    )


# ─── URL guard ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "http://localhost:8000/admin",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "https://",
    ],
)
def test_validate_public_url_rejects_internal_and_non_http(url):
    with pytest.raises(ToolError) as exc:
        validate_public_url(url)
    assert exc.value.retryable is False


def test_validate_public_url_accepts_public_urls():
    validate_public_url("https://jobs.example.com/posting/42?ref=x")
    validate_public_url("http://93.184.216.34/")


# ─── fetch_page ──────────────────────────────────────────────────────────────


def transport(status=200, body="", exc: Exception | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if exc is not None:
            raise exc
        return httpx.Response(status, text=body, request=request)

    return httpx.MockTransport(handler)


async def test_fetch_page_strips_markup_and_noise():
    html = """
    <html><head><style>.x{}</style><script>track()</script></head>
    <body><nav>Menu</nav><h1>Senior Engineer</h1><p>Python and Postgres.</p>
    <footer>Footer links</footer></body></html>
    """
    text = await fetch_page("https://jobs.example.com/1", transport=transport(body=html))
    assert "Senior Engineer" in text
    assert "Python and Postgres." in text
    for noise in ("track()", "Menu", "Footer links", ".x{}"):
        assert noise not in text


async def test_fetch_page_truncates_long_pages():
    html = "<p>" + "word " * 5000 + "</p>"
    text = await fetch_page("https://jobs.example.com/1", transport=transport(body=html))
    assert len(text) < MAX_PAGE_CHARS + 100
    assert text.endswith("[Content truncated — page was too long]")


@pytest.mark.parametrize("status,retryable", [(403, False), (404, False), (429, True), (503, True)])
async def test_fetch_page_classifies_http_errors(status, retryable):
    with pytest.raises(ToolError) as exc:
        await fetch_page("https://jobs.example.com/1", transport=transport(status=status))
    assert exc.value.status_code == status
    assert exc.value.retryable is retryable


async def test_fetch_page_network_errors_are_retryable():
    with pytest.raises(ToolError) as exc:
        await fetch_page(
            "https://jobs.example.com/1", transport=transport(exc=httpx.ConnectError("boom"))
        )
    assert exc.value.retryable is True


async def test_fetch_page_empty_page_is_not_retryable():
    with pytest.raises(ToolError) as exc:
        await fetch_page("https://jobs.example.com/1", transport=transport(body="<html></html>"))
    assert exc.value.retryable is False


# ─── tavily_search ───────────────────────────────────────────────────────────


class FakeTavily:
    response: dict = {}
    error: Exception | None = None

    def __init__(self, api_key=None):
        pass

    async def search(self, query, max_results):
        if FakeTavily.error is not None:
            raise FakeTavily.error
        return FakeTavily.response


@pytest.fixture
def fake_tavily(monkeypatch):
    FakeTavily.response = {}
    FakeTavily.error = None
    monkeypatch.setattr(tools, "AsyncTavilyClient", FakeTavily)
    return FakeTavily


async def test_tavily_search_maps_results(fake_tavily):
    fake_tavily.response = {
        "results": [
            {"title": "Overview", "url": "https://a.example", "content": "Founded 2019"},
            {"title": None, "url": None, "content": None},
        ]
    }
    hits = await tavily_search("exampleco", 5)
    assert hits[0] == SearchHit("Overview", "https://a.example", "Founded 2019")
    assert hits[1] == SearchHit("N/A", "", "")


async def test_tavily_bad_key_is_not_retryable(fake_tavily):
    fake_tavily.error = InvalidAPIKeyError()
    with pytest.raises(ToolError) as exc:
        await tavily_search("exampleco", 5)
    assert exc.value.retryable is False


async def test_tavily_other_errors_are_retryable(fake_tavily):
    fake_tavily.error = httpx.ReadTimeout("slow")
    with pytest.raises(ToolError) as exc:
        await tavily_search("exampleco", 5)
    assert exc.value.retryable is True


def test_format_search_hits():
    assert format_search_hits([]) == "No results found."
    text = format_search_hits([SearchHit("T", "https://a.example", "C")])
    assert text == "Title: T\nURL: https://a.example\nContent: C\n"


# ─── ToolExecutor ────────────────────────────────────────────────────────────


async def test_executor_fetch_success_records_source():
    result = await executor().execute("fetch_url", {"url": "https://jobs.example/1"})
    assert not result.is_error
    assert result.content == "page https://jobs.example/1"
    assert result.sources == {"https://jobs.example/1": "page https://jobs.example/1"}
    assert result.attempts == 1


async def test_executor_retries_transient_fetch_errors(spans):
    calls = 0

    async def fetch(url):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ToolError("HTTP error 503", retryable=True, status_code=503)
        return "finally"

    result = await executor(fetch=fetch).execute("fetch_url", {"url": "https://jobs.example/1"})
    assert result.content == "finally"
    assert result.attempts == 3
    (span,) = by_name(spans, "execute_tool fetch_url")
    assert span.attributes["tool.attempts"] == 3
    assert [e.name for e in span.events] == ["retry", "retry"]


async def test_executor_falls_back_to_search_when_fetch_is_blocked(spans):
    async def fetch(url):
        raise ToolError("HTTP error 403", retryable=False, status_code=403)

    searched = []

    async def search(query, max_results):
        searched.append(query)
        return [SearchHit("Posting", "https://index.example/job", "Senior Engineer")]

    result = await executor(fetch=fetch, search=search).execute(
        "fetch_url", {"url": "https://jobs.example/1"}, call_id="toolu_1"
    )
    assert not result.is_error
    assert result.fallback == "search_web"
    assert searched == ["https://jobs.example/1"]
    assert "Fell back to search_web" in result.content
    assert result.sources == {"https://index.example/job": "Posting\nSenior Engineer"}
    (span,) = by_name(spans, "execute_tool fetch_url")
    assert span.attributes["tool.fallback"] == "search_web"
    assert span.attributes["gen_ai.tool.call.id"] == "toolu_1"
    assert "tool_fallback" in [e.name for e in span.events]


async def test_executor_timeouts_retry_then_fall_back():
    async def hang(url):
        await asyncio.sleep(10)

    result = await executor(fetch=hang, timeout=0.01).execute(
        "fetch_url", {"url": "https://jobs.example/1"}
    )
    assert result.fallback == "search_web"
    assert "timed out" in result.content
    assert result.attempts == 4  # 3 fetch attempts + 1 search


async def test_executor_reports_error_when_fallback_fails_too(spans):
    async def fetch(url):
        raise ToolError("HTTP error 403", retryable=False)

    async def search(query, max_results):
        raise ToolError("Search rejected", retryable=False)

    result = await executor(fetch=fetch, search=search).execute(
        "fetch_url", {"url": "https://jobs.example/1"}
    )
    assert result.is_error
    assert "fallback search also failed" in result.content
    (span,) = by_name(spans, "execute_tool fetch_url")
    assert span.status.is_ok is False


async def test_executor_clamps_max_results():
    seen = []

    async def search(query, max_results):
        seen.append(max_results)
        return []

    ex = executor(search=search)
    await ex.execute("search_web", {"query": "a", "max_results": 50})
    await ex.execute("search_web", {"query": "a", "max_results": 0})
    await ex.execute("search_web", {"query": "a", "max_results": "lots"})
    assert seen == [10, 1, 5]


@pytest.mark.parametrize(
    "name,tool_input",
    [("fetch_url", {}), ("search_web", {"query": "  "}), ("delete_files", {"path": "/"})],
)
async def test_executor_rejects_bad_input_and_unknown_tools(name, tool_input):
    result = await executor().execute(name, tool_input)
    assert result.is_error


async def test_executor_lets_run_timeout_propagate():
    clock_now = [0.0]
    deadline = Deadline(1.0, clock=lambda: clock_now[0])
    clock_now[0] = 2.0
    with pytest.raises(RunTimeoutError):
        await executor().execute("fetch_url", {"url": "https://jobs.example/1"}, deadline)
