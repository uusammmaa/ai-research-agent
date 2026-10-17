"""
Mocked orchestrator-logic eval suite.

These cases never touch the network or the real Anthropic API. They script
the sequence of Claude responses and tool outcomes, then assert the agentic
loop in agent/orchestrator.py behaves correctly: JSON parsing (clean / fenced
/ prose-wrapped), validation with repair-on-failure, max_iterations and
timeout guards, unexpected and refusal stop reasons, LLM and tool retries,
the fallback model, and the fetch -> search fallback.
"""

import asyncio
import json

from agent.models import AgentStep
from agent.tools import SearchHit, ToolError
from evals.fakes import (
    CANNED_TOOLS,
    HANG,
    JOB_PAGE,
    JOB_URL,
    SEARCH_HITS,
    VALID_BRIEF,
    FakeResponse,
    Seq,
    by_model,
    make_agent,
    overloaded_error,
    refusal_response,
    research_turns,
    text_response,
    tool_use_response,
)
from evals.scoring import is_valid_research_brief


async def _collect(agent, job_url: str) -> list[dict]:
    steps: list[AgentStep] = []
    async for step in agent.run(job_url):
        steps.append(step)
    return [json.loads(s.model_dump_json(exclude_none=True)) for s in steps]


def _run_case(script, canned_tools, job_url=JOB_URL, **kwargs):
    agent, fake = make_agent(script, canned_tools, **kwargs)
    steps = asyncio.run(_collect(agent, job_url))
    return steps, agent.state, fake


def _find(steps, type_):
    return [s for s in steps if s["type"] == type_]


def _outcomes(steps) -> list[str]:
    """Step types other than the trailing run_summary."""
    return [s["type"] for s in steps if s["type"] != "run_summary"]


def _run_metrics(state) -> dict:
    return {
        "llm_calls": state.llm_calls,
        "llm_attempts": state.llm_attempts,
        "tool_calls": state.tool_calls,
        "tool_attempts": state.tool_attempts,
        "repairs": state.repairs,
        "models_used": state.models_used,
        "input_tokens": state.usage.input_tokens,
        "output_tokens": state.usage.output_tokens,
        "cost_usd": round(state.usage.cost_usd, 6),
        "cost_by_model": {m: round(u.cost_usd, 6) for m, u in state.usage.by_model.items()},
    }


def _score_success_case(steps, state, name: str, extra_checks: dict | None = None) -> dict:
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
    checks.update(extra_checks or {})

    passed = all(checks.values())
    return {
        "name": name,
        "expected": "complete",
        "passed": passed,
        "checks": checks,
        "steps": len(steps),
        **_run_metrics(state),
    }


def _score_error_case(
    steps, state, name: str, message_substring: str, extra_checks: dict | None = None
) -> dict:
    error_steps = _find(steps, "error")
    checks = {
        "errored": len(error_steps) == 1,
        "message_matches": bool(error_steps)
        and message_substring.lower() in error_steps[0]["message"].lower(),
        "error_is_final_outcome": _outcomes(steps)[-1:] == ["error"],
    }
    checks.update(extra_checks or {})
    passed = all(checks.values())
    return {
        "name": name,
        "expected": "error",
        "passed": passed,
        "checks": checks,
        "steps": len(steps),
        **_run_metrics(state),
    }


# ─── Parsing ─────────────────────────────────────────────────────────────────


def case_happy_path():
    script = [*research_turns(), text_response(json.dumps(VALID_BRIEF))]
    steps, state, _ = _run_case(script, CANNED_TOOLS)
    return _score_success_case(steps, state, "happy_path_clean_json")


def case_json_in_markdown_fence():
    fenced = "```json\n" + json.dumps(VALID_BRIEF) + "\n```"
    steps, state, _ = _run_case([*research_turns(), text_response(fenced)], CANNED_TOOLS)
    return _score_success_case(steps, state, "json_wrapped_in_markdown_fence")


def case_json_with_preamble_text():
    prose = (
        "Here is the research brief you requested:\n\n"
        + json.dumps(VALID_BRIEF)
        + "\n\nLet me know if you need anything else!"
    )
    steps, state, _ = _run_case([*research_turns(), text_response(prose)], CANNED_TOOLS)
    return _score_success_case(steps, state, "json_with_prose_preamble")


# ─── Validation and repair ───────────────────────────────────────────────────


def case_malformed_repair_exhausted():
    # Claude never produces JSON: two repair attempts, then a clean error.
    script = [
        *research_turns(),
        text_response("I looked into it but couldn't find enough information to produce a brief."),
    ]
    steps, state, _ = _run_case(script, CANNED_TOOLS)
    return _score_error_case(
        steps,
        state,
        "malformed_no_json_repair_exhausted",
        "failed to parse",
        {"two_repairs_attempted": len(_find(steps, "repair")) == 2},
    )


def case_missing_field_repaired():
    incomplete = {k: v for k, v in VALID_BRIEF.items() if k != "sources"}
    script = [
        *research_turns(),
        text_response(json.dumps(incomplete)),
        text_response(json.dumps(VALID_BRIEF)),
    ]
    steps, state, fake = _run_case(script, CANNED_TOOLS)
    repair_prompt = fake.messages.requests[-1]["messages"][-1]["content"]
    return _score_success_case(
        steps,
        state,
        "missing_field_repaired",
        {
            "one_repair": state.repairs == 1,
            "repair_prompt_names_field": "sources" in repair_prompt,
        },
    )


def case_unretrieved_citation_repaired():
    # The first brief cites a page no tool call returned; the grounding guard
    # rejects it and the repaired brief drops the claim.
    hallucinated = json.loads(json.dumps(VALID_BRIEF))
    hallucinated["sources"].append("https://made-up.example/press")
    hallucinated["claims"].append(
        {
            "statement": "Acme Corp is profitable.",
            "source": "https://made-up.example/press",
            "quote": "Acme Corp is profitable",
        }
    )
    script = [
        *research_turns(),
        text_response(json.dumps(hallucinated)),
        text_response(json.dumps(VALID_BRIEF)),
    ]
    steps, state, _ = _run_case(script, CANNED_TOOLS)
    repair_steps = _find(steps, "repair")
    return _score_success_case(
        steps,
        state,
        "unretrieved_citation_repaired",
        {
            "guard_flagged_url": bool(repair_steps)
            and "made-up.example" in repair_steps[0]["output"],
        },
    )


# ─── Loop guards ─────────────────────────────────────────────────────────────


def case_max_iterations_exceeded():
    # Claude never stops asking for tools. With max_iterations=3 the loop
    # must yield exactly one error step citing the limit.
    script = [tool_use_response([("search_web", {"query": f"probe {i}"})]) for i in range(10)]
    steps, state, _ = _run_case(script, {"search_web": []}, max_iterations=3)
    return _score_error_case(
        steps,
        state,
        "max_iterations_exceeded",
        "max iterations",
        {"three_llm_calls": state.llm_calls == 3},
    )


def case_run_timeout():
    # The second LLM call hangs past the run budget. The deadline caps the
    # per-call timeout, so the run ends instead of retrying forever.
    script = [tool_use_response([("fetch_url", {"url": JOB_URL})]), HANG]
    steps, state, _ = _run_case(script, CANNED_TOOLS, run_timeout_seconds=0.3, llm_timeout=10)
    return _score_error_case(steps, state, "run_timeout", "timed out")


def case_unexpected_stop_reason():
    steps, state, _ = _run_case([FakeResponse(stop_reason="max_tokens", content=[])], {})
    return _score_error_case(steps, state, "unexpected_stop_reason", "unexpected stop reason")


def case_refusal():
    steps, state, _ = _run_case([refusal_response("cyber")], {})
    return _score_error_case(steps, state, "refusal_stop_reason", "declined")


# ─── LLM retries and fallback ────────────────────────────────────────────────


def case_llm_transient_error_retried():
    # One 529 Overloaded, then the same request succeeds.
    script = [overloaded_error(), *research_turns(), text_response(json.dumps(VALID_BRIEF))]
    steps, state, _ = _run_case(script, CANNED_TOOLS)
    return _score_success_case(
        steps,
        state,
        "llm_transient_error_retried",
        {
            "one_retry": state.llm_attempts == state.llm_calls + 1,
            "stayed_on_primary": state.models_used == ["claude-opus-5-5"],
        },
    )


def case_llm_fallback_model():
    # The primary model is overloaded for the whole run; after its retries
    # are spent the run continues on the fallback model.
    turns = iter([*research_turns(), text_response(json.dumps(VALID_BRIEF))])
    script = [
        by_model(
            {
                "claude-opus-5-5": overloaded_error,
                "claude-sonnet-5-5": lambda: next(turns),
            }
        )
    ]
    steps, state, _ = _run_case(script, CANNED_TOOLS)
    return _score_success_case(
        steps,
        state,
        "llm_fallback_model",
        {
            "fell_back": state.fell_back,
            "finished_on_fallback": state.models_used == ["claude-sonnet-5-5"],
        },
    )


# ─── Tool retries and fallback ───────────────────────────────────────────────


def case_tool_transient_error_retried():
    canned = {
        "fetch_url": Seq(ToolError("HTTP error 503", retryable=True, status_code=503), JOB_PAGE),
        "search_web": SEARCH_HITS,
    }
    script = [*research_turns(), text_response(json.dumps(VALID_BRIEF))]
    steps, state, _ = _run_case(script, canned)
    return _score_success_case(
        steps,
        state,
        "tool_transient_error_retried",
        {"fetch_retried_once": state.tool_attempts == state.tool_calls + 1},
    )


def case_tool_error_recovery():
    # fetch_url is blocked (403, not retryable). The tool layer falls back to
    # searching for the URL, and Claude completes from the search results.
    # The posting was never fetched, so the brief may only cite JOB_URL
    # because the fallback search returned it.
    job_hit = SearchHit("Acme Corp Senior Backend Engineer", JOB_URL, JOB_PAGE)
    canned = {
        "fetch_url": ToolError("HTTP error 403 fetching", retryable=False, status_code=403),
        "search_web": [job_hit, *SEARCH_HITS],
    }
    script = [*research_turns(), text_response(json.dumps(VALID_BRIEF))]
    steps, state, _ = _run_case(script, canned)
    results = _find(steps, "tool_result")
    return _score_success_case(
        steps,
        state,
        "tool_error_recovery_fetch_falls_back_to_search",
        {"fallback_reported": bool(results) and "fell back to search_web" in results[0]["message"]},
    )


def case_tool_timeout_is_error_result():
    # Every fetch and the fallback search hang: the tool result comes back
    # as an error (not an exception) and Claude can still answer.
    canned = {"fetch_url": HANG, "search_web": HANG}
    script = [
        tool_use_response([("fetch_url", {"url": JOB_URL})]),
        text_response("No data available."),
    ]
    steps, state, fake = _run_case(script, canned, tool_timeout=0.05, max_repair_attempts=0)
    tool_result_block = fake.messages.requests[1]["messages"][-1]["content"][0]
    return _score_error_case(
        steps,
        state,
        "tool_timeout_returns_error_result",
        "failed to parse",
        {"tool_result_is_error": tool_result_block.get("is_error") is True},
    )


ALL_CASES = [
    case_happy_path,
    case_json_in_markdown_fence,
    case_json_with_preamble_text,
    case_malformed_repair_exhausted,
    case_missing_field_repaired,
    case_unretrieved_citation_repaired,
    case_max_iterations_exceeded,
    case_run_timeout,
    case_unexpected_stop_reason,
    case_refusal,
    case_llm_transient_error_retried,
    case_llm_fallback_model,
    case_tool_transient_error_retried,
    case_tool_error_recovery,
    case_tool_timeout_is_error_result,
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
                    "checks": {"exception": f"{type(e).__name__}: {e}"},
                    "steps": 0,
                }
            )
    return results
