"""Turning Claude's final text into a validated ResearchBrief."""

import json
import re
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from agent.models import ResearchBrief


class BriefValidationError(ValueError):
    """The final answer could not be turned into a valid, grounded brief.

    problems is a list of short, model-readable strings that the repair
    prompt sends back to Claude.
    """

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def extract_json(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate

    start = 0
    while True:
        idx = text.find("{", start)
        if idx == -1:
            break
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, idx)
            if isinstance(obj, dict):
                return json.dumps(obj)
        except json.JSONDecodeError:
            pass
        start = idx + 1

    return text


def normalize_url(url: str) -> str:
    """Canonical form for comparing URLs: lowercase scheme and host, no
    fragment, no trailing slash."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _format_validation_error(err: ValidationError) -> list[str]:
    problems = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "brief"
        problems.append(f"{loc}: {e['msg']}")
    return problems


def parse_brief(text: str, retrieved_urls: set[str] | None = None) -> ResearchBrief:
    """Parse and validate the final answer.

    Beyond the schema, checks that every claim cites a URL that a tool call
    in this run actually retrieved. That is cheap, deterministic, and catches
    the most common grounding failure: a citation to a page the agent never
    read. Whether the page supports the claim is left to the eval suite.
    """
    try:
        data = json.loads(extract_json(text))
    except (json.JSONDecodeError, ValueError) as e:
        raise BriefValidationError([f"response is not a JSON object ({e})"]) from e
    if not isinstance(data, dict):
        raise BriefValidationError(["response must be a JSON object"])

    try:
        brief = ResearchBrief.model_validate(data)
    except ValidationError as e:
        raise BriefValidationError(_format_validation_error(e)) from e

    if retrieved_urls is not None:
        known = {normalize_url(u) for u in retrieved_urls}
        unretrieved = sorted(
            {c.source for c in brief.claims if normalize_url(c.source) not in known}
        )
        if unretrieved:
            raise BriefValidationError(
                [
                    "claims cite URLs that no tool call in this run retrieved: "
                    f"{unretrieved}. Cite only pages you fetched or that appeared in search "
                    "results, or drop the claim."
                ]
            )
    return brief
