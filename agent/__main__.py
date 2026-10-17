"""Run the agent from the command line.

    python -m agent https://example.com/jobs/123

Prints each step as it happens, then the brief, the run summary and (with
the default console exporter) the trace tree.
"""

import argparse
import asyncio
import json
import logging

from agent.orchestrator import ResearchAgent
from agent.telemetry import setup_tracing, shutdown_tracing


async def _main(job_url: str) -> int:
    agent = ResearchAgent()
    async for step in agent.run(job_url):
        if step.type == "complete":
            print(json.dumps(json.loads(step.output), indent=2))
        elif step.type == "run_summary":
            print(f"\n[run_summary] {step.message}")
            print(json.dumps(step.data, indent=2))
        else:
            print(f"[{step.type}] {step.message}")
    return 0 if agent.state and agent.state.status == "complete" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("job_url")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_tracing()
    try:
        code = asyncio.run(_main(args.job_url))
    finally:
        shutdown_tracing()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
