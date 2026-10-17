"""
Live end-to-end eval suite.

Runs the real agentic loop (real Anthropic + Tavily API calls, real HTTP
fetches) against a handful of currently-live remote job postings, sourced at
run time from RemoteOK's public API so the harness doesn't rot on stale URLs.
"""

import asyncio
import json
import time

import httpx

from agent import orchestrator
from config import settings
from evals.grounding import grounding_report
from evals.judge import judge_claims
from evals.scoring import is_valid_research_brief


def get_live_job_urls(n: int = 3) -> list[dict]:
    """Pull n current job postings from RemoteOK's public API."""
    resp = httpx.get(
        "https://remoteok.com/api",
        headers={"User-Agent": "Mozilla/5.0 (compatible; EvalHarness/1.0)"},
        timeout=15,
    )
    resp.raise_for_status()
    listings = [item for item in resp.json() if item.get("url") and item.get("company")]
    picked = listings[1 : 1 + n]  # index 0 is API metadata, not a job
    return [
        {
            "url": item["url"],
            "company": item["company"].strip(),
            "position": item.get("position", ""),
        }
        for item in picked
    ]


async def _run_one(job_url: str) -> dict:
    """Runs the real agent. iteration_count is LLM calls (RunState.llm_calls)
    rather than tool_call steps: one response can hold several tool_use
    blocks, so tool-call count alone overstates iterations used."""
    agent = orchestrator.ResearchAgent()
    steps = []
    start = time.monotonic()
    async for step in agent.run(job_url):
        steps.append(json.loads(step.model_dump_json(exclude_none=True)))
    elapsed = time.monotonic() - start
    return {
        "steps": steps,
        "elapsed_seconds": round(elapsed, 1),
        "iteration_count": agent.state.llm_calls,
        "state": agent.state,
    }


def _find(steps, type_):
    return [s for s in steps if s["type"] == type_]


LIVE_GROUNDEDNESS_THRESHOLD = 0.8
LIVE_LATENCY_BUDGET_SECONDS = 180


def score_live_run(name: str, job: dict, run: dict, judge: bool = False) -> dict:
    steps = run["steps"]
    state = run["state"]
    complete_steps = _find(steps, "complete")
    error_steps = _find(steps, "error")
    tool_calls = _find(steps, "tool_call")

    checks = {"completed": len(complete_steps) == 1 and not error_steps}

    brief = None
    valid_schema = False
    sources_present = False
    if complete_steps:
        try:
            brief = json.loads(complete_steps[0]["output"])
            valid_schema = is_valid_research_brief(brief)
            sources_present = len(brief.get("sources", [])) > 0
        except Exception:
            pass

    grounding = grounding_report(brief, state.corpus) if brief else None
    checks["valid_schema"] = valid_schema
    checks["sources_present"] = sources_present
    checks["claims_cite_retrieved_sources"] = (
        bool(grounding) and grounding["citation_accuracy"] == 1.0
    )
    checks["claims_grounded"] = (
        bool(grounding)
        and grounding["groundedness"] is not None
        and grounding["groundedness"] >= LIVE_GROUNDEDNESS_THRESHOLD
    )
    checks["first_tool_is_fetch_url"] = bool(tool_calls) and tool_calls[0]["tool"] == "fetch_url"
    checks["used_search"] = any(t["tool"] == "search_web" for t in tool_calls)
    checks["within_iteration_budget"] = run["iteration_count"] <= settings.max_iterations
    checks["latency_ok"] = run["elapsed_seconds"] <= LIVE_LATENCY_BUDGET_SECONDS

    judged = judge_claims(brief["claims"], state.corpus) if (judge and brief) else None
    error_message = error_steps[0]["message"] if error_steps else None

    return {
        "name": name,
        "url": job["url"],
        "expected": "complete",
        "passed": all(checks.values()),
        "checks": checks,
        "steps": len(steps),
        "tool_call_count": len(tool_calls),
        "iteration_count": run["iteration_count"],
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
        "trace_id": state.trace_id,
        "elapsed_seconds": run["elapsed_seconds"],
        "error_message": error_message,
        "grounding": grounding,
        "judge": judged,
        "brief": brief,
    }


def run_all(n: int = 3, judge: bool = False) -> list[dict]:
    jobs = get_live_job_urls(n)
    results = []
    # Names are positional so committed tables never carry employer names;
    # the URL stays in results.json, which is gitignored.
    for i, job in enumerate(jobs, start=1):
        name = f"live::job-{i}"
        try:
            run = asyncio.run(_run_one(job["url"]))
            results.append(score_live_run(name, job, run, judge=judge))
        except Exception as e:
            results.append(
                {
                    "name": name,
                    "url": job["url"],
                    "expected": "complete",
                    "passed": False,
                    "checks": {"exception": str(e)},
                    "steps": 0,
                    "tool_call_count": 0,
                    "iteration_count": 0,
                    "elapsed_seconds": None,
                    "error_message": str(e),
                    "brief": None,
                }
            )
    return results
