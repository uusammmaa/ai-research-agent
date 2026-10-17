from typing import Any

import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient

from config import settings

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


# ─── Tool Execution ───────────────────────────────────────────────────────────
# These are the actual Python functions that run when Claude calls a tool.


async def fetch_url(url: str) -> str:
    """Fetch a URL and return clean readable text."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Trim to avoid exceeding context window
        if len(text) > 8000:
            text = text[:8000] + "\n\n[Content truncated — page was too long]"

        return text or "No readable content found at this URL."

    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily and return formatted results."""
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=max_results)

        results = []
        for r in response.get("results", []):
            results.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', 'N/A')}\n"
                f"Content: {r.get('content', 'N/A')}\n"
            )

        return "\n---\n".join(results) if results else "No results found."

    except Exception as e:
        return f"Search error: {str(e)}"


async def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Route tool calls to the right function."""
    if tool_name == "fetch_url":
        return await fetch_url(tool_input["url"])
    elif tool_name == "search_web":
        max_results = min(max(tool_input.get("max_results", 5), 1), 10)
        return search_web(query=tool_input["query"], max_results=max_results)
    else:
        return f"Unknown tool: {tool_name}"
