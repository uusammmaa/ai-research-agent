"""A scripted run with tracing on, for seeing the span tree without API keys.

    python -m evals.demo

Claude's responses and tool outcomes are scripted (evals/fakes.py); the
first LLM call gets a 529, the first page fetch a 503, and the first brief is
missing a field, so the trace shows a retry on each layer and a repair.
Durations are near zero because nothing goes over the network.
"""

import asyncio
import json
import logging

from agent.telemetry import setup_tracing
from agent.tools import ToolError
from evals.fakes import (
    JOB_PAGE,
    JOB_URL,
    SEARCH_HITS,
    VALID_BRIEF,
    Seq,
    make_agent,
    overloaded_error,
    research_turns,
    text_response,
)


async def main() -> None:
    incomplete = {k: v for k, v in VALID_BRIEF.items() if k != "sources"}
    agent, _ = make_agent(
        [
            overloaded_error(),
            *research_turns(),
            text_response(json.dumps(incomplete)),
            text_response(json.dumps(VALID_BRIEF)),
        ],
        {
            "fetch_url": Seq(
                ToolError("HTTP error 503", retryable=True, status_code=503), JOB_PAGE
            ),
            "search_web": SEARCH_HITS,
        },
    )
    async for step in agent.run(JOB_URL):
        if step.type == "run_summary":
            print(f"[run_summary] {step.message}")
            print(json.dumps(step.data, indent=2))
        elif step.type == "complete":
            print(
                f"[complete] {step.message}: {json.loads(step.output)['role']} at {json.loads(step.output)['company']}"
            )
        else:
            print(f"[{step.type}] {step.message}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    setup_tracing("console")
    asyncio.run(main())
