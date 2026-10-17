import asyncio
import json

import pytest

from agent.orchestrator import ResearchAgent, run_agent
from agent.reliability import RunTimeoutError
from agent.tools import ToolExecutor, ToolResult
from evals.fakes import (
    CANNED_TOOLS,
    JOB_URL,
    VALID_BRIEF,
    api_status_error,
    make_agent,
    research_turns,
    text_response,
    tool_use_response,
)
from tests.conftest import by_name


async def collect(agent: ResearchAgent, url: str = JOB_URL):
    return [step async for step in agent.run(url)]


def types(steps):
    return [s.type for s in steps]


async def test_happy_path_streams_steps_and_summary():
    agent, _ = make_agent([*research_turns(), text_response(json.dumps(VALID_BRIEF))], CANNED_TOOLS)
    steps = await collect(agent)
    assert types(steps) == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "complete",
        "run_summary",
    ]
    assert json.loads(steps[4].output)["company"] == "Example Co"

    summary = steps[-1].data
    assert summary["status"] == "complete"
    assert summary["llm_calls"] == 3
    assert summary["tool_calls"] == 2
    assert summary["usage"]["by_model"]["claude-opus-5-5"]["calls"] == 3
    assert summary["usage"]["cost_usd"] > 0
    assert len(summary["trace_id"]) == 32
    assert set(agent.state.corpus) == {JOB_URL, *(h.url for h in CANNED_TOOLS["search_web"])}


async def test_parallel_tool_calls_run_concurrently_and_return_in_one_message():
    started, release = [], asyncio.Event()

    async def fetch(url):
        started.append(url)
        if len(started) == 2:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=1)  # deadlocks if run serially
        return f"page {url}"

    async def search(query, max_results):
        return []

    tools = ToolExecutor(fetch=fetch, search=search, timeout=2)
    agent, fake = make_agent(
        [
            tool_use_response(
                [
                    ("fetch_url", {"url": "https://a.example/1"}),
                    ("fetch_url", {"url": "https://b.example/2"}),
                ]
            ),
            text_response("not json"),
        ],
        {},
        max_repair_attempts=0,
    )
    agent.tools = tools
    await collect(agent)

    assert sorted(started) == ["https://a.example/1", "https://b.example/2"]
    tool_message = fake.messages.requests[1]["messages"][-1]
    assert tool_message["role"] == "user"
    assert [b["tool_use_id"] for b in tool_message["content"]] == [
        "tool_fetch_url_0",
        "tool_fetch_url_1",
    ]
    assert [b["content"] for b in tool_message["content"]] == [
        "page https://a.example/1",
        "page https://b.example/2",
    ]


async def test_tool_failures_reach_claude_as_error_results():
    class Failing(ToolExecutor):
        async def execute(self, name, tool_input, deadline=None, call_id=None):
            return ToolResult(content="Search failed: rate limited", is_error=True, attempts=3)

    agent, fake = make_agent(
        [tool_use_response([("search_web", {"query": "x"})]), text_response("nothing")],
        {},
        max_repair_attempts=0,
    )
    agent.tools = Failing()
    steps = await collect(agent)
    block = fake.messages.requests[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert steps[1].message == "search_web failed"
    assert agent.state.tool_attempts == 3


async def test_repair_turns_do_not_consume_the_iteration_budget():
    bad = {k: v for k, v in VALID_BRIEF.items() if k != "claims"}
    agent, fake = make_agent(
        [*research_turns(), text_response(json.dumps(bad)), text_response(json.dumps(VALID_BRIEF))],
        CANNED_TOOLS,
        max_iterations=3,
    )
    steps = await collect(agent)
    assert types(steps)[-3:] == ["repair", "complete", "run_summary"]
    assert agent.state.llm_calls == 4
    repair_prompt = fake.messages.requests[-1]["messages"][-1]["content"]
    assert "claims" in repair_prompt and "failed validation" in repair_prompt


async def test_non_retryable_llm_error_ends_run_with_error():
    agent, _ = make_agent([api_status_error(401, "invalid x-api-key")], {})
    steps = await collect(agent)
    assert types(steps) == ["error", "run_summary"]
    assert "LLM call failed" in steps[0].message
    assert agent.state.status == "error"


async def test_run_timeout_during_parallel_tools_leaves_no_orphan_tasks():
    finished = []

    class OneTimesOut(ToolExecutor):
        async def execute(self, name, tool_input, deadline=None, call_id=None):
            if tool_input["url"].endswith("/1"):
                raise RunTimeoutError("out of time")
            await asyncio.sleep(0.05)
            finished.append(tool_input["url"])
            return ToolResult(content="slow page")

    agent, _ = make_agent(
        [
            tool_use_response(
                [
                    ("fetch_url", {"url": "https://a.example/1"}),
                    ("fetch_url", {"url": "https://b.example/2"}),
                ]
            )
        ],
        {},
    )
    agent.tools = OneTimesOut()
    steps = await collect(agent)
    assert "timed out" in [s for s in steps if s.type == "error"][0].message
    # The slower sibling ran to completion before the run reported the error.
    assert finished == ["https://b.example/2"]


async def test_span_tree_covers_every_step_tool_and_llm_call(spans):
    agent, _ = make_agent([*research_turns(), text_response(json.dumps(VALID_BRIEF))], CANNED_TOOLS)
    await collect(agent)

    (root,) = by_name(spans, "invoke_agent")
    steps = by_name(spans, "agent.step")
    chats = by_name(spans, "chat ")
    tool_spans = by_name(spans, "execute_tool")
    (validate,) = by_name(spans, "agent.validate_brief")

    assert len(steps) == len(chats) == 3
    assert [s.name for s in tool_spans] == ["execute_tool fetch_url", "execute_tool search_web"]
    assert {s.parent.span_id for s in steps} == {root.context.span_id}
    step_ids = {s.context.span_id for s in steps}
    assert all(s.parent.span_id in step_ids for s in [*chats, *tool_spans, validate])
    assert {s.context.trace_id for s in spans.get_finished_spans()} == {root.context.trace_id}

    assert root.attributes["agent.status"] == "complete"
    assert root.attributes["agent.llm_calls"] == 3
    assert root.attributes["gen_ai.usage.input_tokens"] == sum(
        c.attributes["gen_ai.usage.input_tokens"] for c in chats
    )
    assert root.attributes["agent.cost_usd"] == pytest.approx(
        sum(c.attributes["llm.cost_usd"] for c in chats), abs=1e-6
    )
    assert root.attributes["agent.cost_usd.claude-opus-5-5"] > 0
    assert validate.attributes["agent.brief_valid"] is True


async def test_root_span_ends_when_consumer_stops_early(spans):
    agent, _ = make_agent([*research_turns(), text_response(json.dumps(VALID_BRIEF))], CANNED_TOOLS)
    gen = agent.run(JOB_URL)
    await gen.__anext__()
    await gen.aclose()
    (root,) = by_name(spans, "invoke_agent")
    assert root.attributes["agent.status"] == "cancelled"


async def test_run_agent_emits_ndjson():
    agent, _ = make_agent([*research_turns(), text_response(json.dumps(VALID_BRIEF))], CANNED_TOOLS)
    lines = [line async for line in run_agent(JOB_URL, agent)]
    assert all(line.endswith("\n") for line in lines)
    parsed = [json.loads(line) for line in lines]
    assert [p["type"] for p in parsed][-2:] == ["complete", "run_summary"]
    # Steps without structured data omit the field on the wire.
    assert "data" not in parsed[0]
