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


def score_live_run(job: dict, run: dict) -> dict:
    steps = run["steps"]
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

    checks["valid_schema"] = valid_schema
    checks["sources_present"] = sources_present
    checks["first_tool_is_fetch_url"] = bool(tool_calls) and tool_calls[0]["tool"] == "fetch_url"
    checks["used_search"] = any(t["tool"] == "search_web" for t in tool_calls)
    checks["within_iteration_budget"] = run["iteration_count"] <= settings.max_iterations
    checks["latency_ok"] = run["elapsed_seconds"] <= 120

    error_message = error_steps[0]["message"] if error_steps else None

    return {
        "name": f"live::{job['company']}",
        "url": job["url"],
        "expected": "complete",
        "passed": all(checks.values()),
        "checks": checks,
        "steps": len(steps),
        "tool_call_count": len(tool_calls),
        "iteration_count": run["iteration_count"],
        "elapsed_seconds": run["elapsed_seconds"],
        "error_message": error_message,
        "brief": brief,
    }


def run_all(n: int = 3) -> list[dict]:
    jobs = get_live_job_urls(n)
    results = []
    for job in jobs:
        try:
            run = asyncio.run(_run_one(job["url"]))
            results.append(score_live_run(job, run))
        except Exception as e:
            results.append(
                {
                    "name": f"live::{job['company']}",
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
