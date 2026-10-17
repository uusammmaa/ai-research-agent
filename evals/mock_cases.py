"""
Mocked orchestrator-logic eval suite.

These cases never touch the network or the real Anthropic API. They script
the sequence of Claude responses (via FakeResponse) and tool results, then
assert the agentic loop in agent/orchestrator.py behaves correctly against
each edge case called out in the project's own troubleshooting table:
JSON parsing (clean / fenced / prose-wrapped / malformed / missing field),
max_iterations enforcement, unexpected stop_reason, and tool-error recovery.
"""

import asyncio
import json
from unittest.mock import patch

from agent import orchestrator
from evals.fakes import (
    VALID_BRIEF,
    FakeResponse,
    make_create_stub,
    make_execute_tool_stub,
    text_response,
    tool_use_response,
)
from evals.scoring import is_valid_research_brief


async def _collect_steps(job_url: str) -> list[dict]:
    steps = []
    async for line in orchestrator.run_agent(job_url):
        steps.append(json.loads(line))
    return steps


def _run_case(responses, tool_results, job_url="https://example.com/job/123", max_iterations=None):
    create_stub = make_create_stub(responses)
    execute_stub = make_execute_tool_stub(tool_results)

    patches = [
        patch.object(orchestrator.client.messages, "create", side_effect=create_stub),
        patch.object(orchestrator, "execute_tool", side_effect=execute_stub),
    ]
    if max_iterations is not None:
        patches.append(patch.object(orchestrator.settings, "max_iterations", max_iterations))

    for p in patches:
        p.start()
    try:
        return asyncio.run(_collect_steps(job_url))
    finally:
        for p in reversed(patches):
            p.stop()


def _find(steps, type_):
    return [s for s in steps if s["type"] == type_]


def _score_success_case(steps: dict, name: str) -> dict:
    checks = {}
    complete_steps = _find(steps, "complete")
    error_steps = _find(steps, "error")

    checks["completed"] = len(complete_steps) == 1 and len(error_steps) == 0

    valid_schema = False
    if complete_steps:
        try:
            brief = json.loads(complete_steps[0]["output"])
            valid_schema = is_valid_research_brief(brief)
        except Exception:
            valid_schema = False
    checks["valid_schema"] = valid_schema

    tool_calls = _find(steps, "tool_call")
    checks["first_tool_is_fetch_url"] = (
        (bool(tool_calls) and tool_calls[0]["tool"] == "fetch_url") if tool_calls else True
    )  # no tools called is not a grounding violation by itself

    passed = all(checks.values())
    return {
        "name": name,
        "expected": "complete",
        "passed": passed,
        "checks": checks,
        "steps": len(steps),
    }


def _score_error_case(steps, name: str, message_substring: str) -> dict:
    error_steps = _find(steps, "error")
    checks = {
        "errored": len(error_steps) == 1,
        "message_matches": bool(error_steps)
        and message_substring.lower() in error_steps[0]["message"].lower(),
    }
    passed = all(checks.values())
    return {
        "name": name,
        "expected": "error",
        "passed": passed,
        "checks": checks,
        "steps": len(steps),
    }


def case_happy_path():
    responses = [
        tool_use_response([("fetch_url", {"url": "https://example.com/job/123"})]),
        tool_use_response([("search_web", {"query": "Acme Corp company overview funding"})]),
        tool_use_response([("search_web", {"query": "Acme Corp news 2025"})]),
        text_response(json.dumps(VALID_BRIEF)),
    ]
    tool_results = {
        "fetch_url": "Senior Backend Engineer at Acme Corp. Remote. Python, FastAPI, PostgreSQL required.",
        "search_web": "Title: Acme Corp Overview\nURL: https://acme.com/about\nContent: Series B fintech, founded 2019.",
    }
    steps = _run_case(responses, tool_results)
    return _score_success_case(steps, "happy_path_clean_json")


def case_json_in_markdown_fence():
    fenced = "```json\n" + json.dumps(VALID_BRIEF) + "\n```"
    responses = [
        tool_use_response([("fetch_url", {"url": "https://example.com/job/123"})]),
        text_response(fenced),
    ]
    tool_results = {"fetch_url": "Senior Backend Engineer at Acme Corp."}
    steps = _run_case(responses, tool_results)
    return _score_success_case(steps, "json_wrapped_in_markdown_fence")


def case_json_with_preamble_text():
    prose = (
        "Here is the research brief you requested:\n\n"
        + json.dumps(VALID_BRIEF)
        + "\n\nLet me know if you need anything else!"
    )
    responses = [
        tool_use_response([("fetch_url", {"url": "https://example.com/job/123"})]),
        text_response(prose),
    ]
    tool_results = {"fetch_url": "Senior Backend Engineer at Acme Corp."}
    steps = _run_case(responses, tool_results)
    return _score_success_case(steps, "json_with_prose_preamble")


def case_malformed_unparseable():
    responses = [
        tool_use_response([("fetch_url", {"url": "https://example.com/job/123"})]),
        text_response("I looked into it but couldn't find enough information to produce a brief."),
    ]
    tool_results = {"fetch_url": "Senior Backend Engineer at Acme Corp."}
    steps = _run_case(responses, tool_results)
    return _score_error_case(steps, "malformed_no_json", "failed to parse")


def case_missing_required_field():
    incomplete = {k: v for k, v in VALID_BRIEF.items() if k != "sources"}
    responses = [
        tool_use_response([("fetch_url", {"url": "https://example.com/job/123"})]),
        text_response(json.dumps(incomplete)),
    ]
    tool_results = {"fetch_url": "Senior Backend Engineer at Acme Corp."}
    steps = _run_case(responses, tool_results)
    return _score_error_case(steps, "missing_required_field", "failed to parse")


def case_max_iterations_exceeded():
    # Claude never stops asking for tools. With max_iterations patched to 3,
    # the loop must yield exactly one error step citing the limit.
    responses = [tool_use_response([("search_web", {"query": f"probe {i}"})]) for i in range(10)]
    tool_results = {"search_web": "No results found."}
    steps = _run_case(responses, tool_results, max_iterations=3)
    return _score_error_case(steps, "max_iterations_exceeded", "max iterations")


def case_unexpected_stop_reason():
    responses = [FakeResponse(stop_reason="max_tokens", content=[])]
    steps = _run_case(responses, {})
    return _score_error_case(steps, "unexpected_stop_reason", "unexpected stop reason")


def case_tool_error_recovery():
    # fetch_url "fails" (as the real tool does: returns an error string rather
    # than raising), Claude recovers per the system prompt's fallback rule and
    # searches instead, then completes successfully.
    responses = [
        tool_use_response([("fetch_url", {"url": "https://example.com/job/123"})]),
        tool_use_response([("search_web", {"query": "site:example.com job/123 posting"})]),
        text_response(json.dumps(VALID_BRIEF)),
    ]
    tool_results = {
        "fetch_url": "Error fetching https://example.com/job/123: 403 Forbidden",
        "search_web": "Title: Acme Corp Senior Backend Engineer\nURL: https://example.com/job/123\nContent: role details...",
    }
    steps = _run_case(responses, tool_results)
    return _score_success_case(steps, "tool_error_recovery")


ALL_CASES = [
    case_happy_path,
    case_json_in_markdown_fence,
    case_json_with_preamble_text,
    case_malformed_unparseable,
    case_missing_required_field,
    case_max_iterations_exceeded,
    case_unexpected_stop_reason,
    case_tool_error_recovery,
]


def run_all() -> list[dict]:
    results = []
    for case_fn in ALL_CASES:
        try:
            results.append(case_fn())
        except Exception as e:
            results.append(
                {
                    "name": case_fn.__name__,
                    "expected": "?",
                    "passed": False,
                    "checks": {"exception": str(e)},
                    "steps": 0,
                }
            )
    return results
