"""
Eval harness entry point for the AI Research Agent.

Usage (from backend/, with venv active):
    python -m evals.run_evals                 # mocked suite only
    python -m evals.run_evals --live           # mocked + live suite
    python -m evals.run_evals --live-only      # live suite only
    python -m evals.run_evals --live -n 5      # live suite with 5 jobs
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals import live_cases, mock_cases

RESULTS_PATH = Path(__file__).parent / "results.json"


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {"total": total, "passed": passed, "failed": total - passed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also run the live suite")
    parser.add_argument("--live-only", action="store_true", help="skip mocked suite, run live only")
    parser.add_argument("-n", type=int, default=3, help="number of live job URLs to test")
    args = parser.parse_args()

    mock_results = [] if args.live_only else mock_cases.run_all()
    live_results = live_cases.run_all(args.n) if (args.live or args.live_only) else []

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mock_suite": {"summary": summarize(mock_results), "results": mock_results},
        "live_suite": {"summary": summarize(live_results), "results": live_results}
        if live_results or args.live or args.live_only
        else None,
    }

    RESULTS_PATH.write_text(json.dumps(report, indent=2))

    print(
        f"\nMocked suite: {report['mock_suite']['summary']['passed']}/{report['mock_suite']['summary']['total']} passed"
    )
    for r in mock_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}")
        if not r["passed"]:
            print(f"         checks: {r['checks']}")

    if report["live_suite"]:
        print(
            f"\nLive suite: {report['live_suite']['summary']['passed']}/{report['live_suite']['summary']['total']} passed"
        )
        for r in live_results:
            status = "PASS" if r["passed"] else "FAIL"
            print(
                f"  [{status}] {r['name']} ({r.get('elapsed_seconds')}s, {r.get('tool_call_count')} tool calls)"
            )
            if not r["passed"]:
                print(f"         checks: {r['checks']}  error: {r.get('error_message')}")

    print(f"\nResults written to {RESULTS_PATH}")

    any_fail = any(not r["passed"] for r in mock_results + live_results)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
