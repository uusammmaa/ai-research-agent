import asyncio
import ipaddress
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tavily import AsyncTavilyClient, BadRequestError, InvalidAPIKeyError, MissingAPIKeyError

from agent.reliability import Deadline, RetryPolicy, retry_async
from config import settings

MAX_PAGE_CHARS = 8000

# ─── Tool Schemas ────────────────────────────────────────────────────────────
# These are what Claude reads to decide which tool to use and how.
# Description quality directly affects agent performance.

TOOL_SCHEMAS = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch and extract the readable text content from a web page URL. "
            "Use this to read job postings, company about pages, or any specific URL. "
            "Returns clean text with HTML removed. "
            "If the page cannot be fetched, the tool automatically falls back to a web "
            "search for that URL and says so in its result. "
            "Do not use for general research — use search_web for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch, must start with http:// or https://",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for information about a topic. "
            "Use this to research companies, find news, look up technologies, "
            "or gather any information not available at a specific URL. "
            "Returns a list of relevant results with titles, URLs, and content snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific. Include company name and year for recent news.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5, max 10.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]


# ─── Result types ────────────────────────────────────────────────────────────


class ToolError(Exception):
    """A tool failure, tagged with whether trying again could help."""

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@dataclass
class SearchHit:
    title: str
    url: str
    content: str


@dataclass
class ToolResult:
    """What the orchestrator gets back from one tool call.

    content goes back to Claude; sources maps every URL the call retrieved to
    the text it retrieved, which is what citation and groundedness checks
    compare claims against.
    """

    content: str
    is_error: bool = False
    sources: dict[str, str] = field(default_factory=dict)
    attempts: int = 1
    fallback: str | None = None
    error: str | None = None


# ─── Raw tool implementations ────────────────────────────────────────────────
# These do one network call each and raise ToolError on failure. Retries,
# timeouts and fallbacks live in ToolExecutor so they apply uniformly.


def validate_public_url(url: str) -> None:
    """Reject non-HTTP schemes and obvious internal targets.

    The URL comes from model output, which can be steered by page content, so
    it is untrusted. This blocks the cheap SSRF cases (localhost, private and
    link-local IP literals). It does not resolve DNS, so a public hostname that
    resolves to a private address is not caught; run the service without
    access to internal networks for that.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError(
            f"Refusing to fetch {url!r}: only http(s) URLs are allowed", retryable=False
        )
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".internal"):
        raise ToolError(f"Refusing to fetch internal host {host!r}", retryable=False)
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        raise ToolError(f"Refusing to fetch non-public address {host!r}", retryable=False)


def _is_retryable_status(status: int) -> bool:
    return status in (408, 425, 429) or status >= 500


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Trim to avoid exceeding context window
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n\n[Content truncated — page was too long]"
    return text


async def fetch_page(url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """Fetch a URL and return clean readable text."""
    validate_public_url(url)
    try:
        async with httpx.AsyncClient(
            timeout=settings.tool_timeout_seconds, follow_redirects=True, transport=transport
        ) as client:
            response = await client.get(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        raise ToolError(
            f"HTTP error {status} fetching {url}",
            retryable=_is_retryable_status(status),
            status_code=status,
        ) from e
    except httpx.TransportError as e:
        # Timeouts, DNS failures, resets: all transient from our point of view.
        raise ToolError(f"Network error fetching {url}: {e!r}", retryable=True) from e

    text = html_to_text(response.text)
    if not text:
        raise ToolError(f"No readable content found at {url}", retryable=False)
    return text


async def tavily_search(query: str, max_results: int = 5) -> list[SearchHit]:
    """Search the web using Tavily."""
    try:
        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        response = await client.search(query=query, max_results=max_results)
    except (InvalidAPIKeyError, MissingAPIKeyError, BadRequestError) as e:
        raise ToolError(f"Search rejected: {type(e).__name__}: {e}", retryable=False) from e
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        raise ToolError(
            f"Search HTTP error {status}",
            retryable=_is_retryable_status(status),
            status_code=status,
        ) from e
    except Exception as e:
        # Rate limits (UsageLimitExceededError) and transport errors.
        raise ToolError(f"Search error: {type(e).__name__}: {e}", retryable=True) from e

    return [
        SearchHit(
            title=r.get("title") or "N/A",
            url=r.get("url") or "",
            content=r.get("content") or "",
        )
        for r in response.get("results", [])
    ]


def format_search_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "No results found."
    return "\n---\n".join(
        f"Title: {h.title}\nURL: {h.url or 'N/A'}\nContent: {h.content}\n" for h in hits
    )


# ─── Tool Execution ───────────────────────────────────────────────────────────

FetchFn = Callable[[str], Awaitable[str]]
SearchFn = Callable[[str, int], Awaitable[list[SearchHit]]]


def _is_retryable_tool_error(exc: BaseException) -> bool:
    if isinstance(exc, ToolError):
        return exc.retryable
    return isinstance(exc, TimeoutError)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timed out"
    return str(exc) or type(exc).__name__


@dataclass
class _Attempts:
    n: int = 0


class ToolExecutor:
    """Runs tools with a per-call timeout, retries with backoff, and a
    fetch_url -> search_web fallback.

    Tool failures never raise out of execute(): they come back as a
    ToolResult with is_error=True so Claude can see what went wrong and adapt.
    Only RunTimeoutError (the whole run is out of time) propagates.
    """

    def __init__(
        self,
        *,
        fetch: FetchFn = fetch_page,
        search: SearchFn = tavily_search,
        retry_policy: RetryPolicy | None = None,
        timeout: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._fetch = fetch
        self._search = search
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=settings.tool_max_attempts)
        self.timeout = timeout if timeout is not None else settings.tool_timeout_seconds
        self._sleep = sleep

    async def _call(
        self, fn: Callable[[], Awaitable[Any]], deadline: Deadline | None, attempts: "_Attempts"
    ) -> Any:
        async def attempt():
            attempts.n += 1
            timeout = deadline.cap(self.timeout) if deadline else self.timeout
            return await asyncio.wait_for(fn(), timeout=timeout)

        return await retry_async(
            attempt,
            policy=self.retry_policy,
            is_retryable=_is_retryable_tool_error,
            deadline=deadline,
            sleep=self._sleep,
        )

    async def _run_search(
        self, query: str, max_results: int, deadline: Deadline | None, attempts: "_Attempts"
    ) -> tuple[str, dict[str, str]]:
        hits = await self._call(lambda: self._search(query, max_results), deadline, attempts)
        sources = {h.url: f"{h.title}\n{h.content}" for h in hits if h.url}
        return format_search_hits(hits), sources

    async def execute(
        self, tool_name: str, tool_input: dict[str, Any], deadline: Deadline | None = None
    ) -> ToolResult:
        """Route tool calls to the right function."""
        if tool_name == "fetch_url":
            return await self._execute_fetch(tool_input, deadline)
        if tool_name == "search_web":
            return await self._execute_search(tool_input, deadline)
        return ToolResult(
            content=f"Unknown tool: {tool_name}", is_error=True, error="unknown_tool", attempts=0
        )

    async def _execute_fetch(self, tool_input: dict[str, Any], deadline: Deadline | None):
        url = tool_input.get("url")
        if not isinstance(url, str) or not url:
            return ToolResult(
                content="fetch_url requires a 'url' string", is_error=True, error="bad_input"
            )
        attempts = _Attempts()
        try:
            text = await self._call(lambda: self._fetch(url), deadline, attempts)
            return ToolResult(content=text, sources={url: text}, attempts=attempts.n)
        except (ToolError, TimeoutError) as e:
            fetch_error = _describe(e)

        # Fallback: job boards often block scrapers, but search engines have
        # indexed the posting. Search for the URL itself.
        try:
            content, sources = await self._run_search(url, 5, deadline, attempts)
        except (ToolError, TimeoutError) as e:
            return ToolResult(
                content=(
                    f"fetch_url failed ({fetch_error}) and the fallback search also failed "
                    f"({_describe(e)}). Try a different source."
                ),
                is_error=True,
                attempts=attempts.n,
                fallback="search_web",
                error=fetch_error,
            )
        return ToolResult(
            content=(
                f"fetch_url failed ({fetch_error}). Fell back to search_web for this URL; "
                f"results below.\n\n{content}"
            ),
            sources=sources,
            attempts=attempts.n,
            fallback="search_web",
            error=fetch_error,
        )

    async def _execute_search(self, tool_input: dict[str, Any], deadline: Deadline | None):
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                content="search_web requires a non-empty 'query' string",
                is_error=True,
                error="bad_input",
            )
        try:
            max_results = int(tool_input.get("max_results", 5))
        except (TypeError, ValueError):
            max_results = 5
        max_results = min(max(max_results, 1), 10)
        attempts = _Attempts()
        try:
            content, sources = await self._run_search(query, max_results, deadline, attempts)
        except (ToolError, TimeoutError) as e:
            return ToolResult(
                content=f"Search failed: {_describe(e)}",
                is_error=True,
                attempts=attempts.n,
                error=_describe(e),
            )
        return ToolResult(content=content, sources=sources, attempts=attempts.n)
