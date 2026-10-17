import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from agent.llm import LLMClient
from agent.models import AgentStep, ResearchBrief
from agent.output import BriefValidationError, parse_brief
from agent.prompts import REPAIR_PROMPT, SYSTEM_PROMPT
from agent.reliability import Deadline, RunTimeoutError
from agent.telemetry import get_tracer
from agent.tools import TOOL_SCHEMAS, ToolExecutor, ToolResult
from agent.usage import RunUsage
from config import settings

log = logging.getLogger(__name__)


def stream_step(step: AgentStep) -> str:
    """Serialise an AgentStep as a newline-delimited JSON string."""
    return json.dumps(step.model_dump(exclude_none=True)) + "\n"


@dataclass
class RunState:
    """Everything one run accumulated. Evals and tests read this after the
    stream ends; the API only sees the streamed steps."""

    job_url: str
    status: str = "running"  # running | complete | error
    llm_calls: int = 0
    llm_attempts: int = 0
    tool_calls: int = 0
    tool_attempts: int = 0
    repairs: int = 0
    models_used: list[str] = field(default_factory=list)
    fell_back: bool = False
    # URL -> text for every page fetched or returned by search in this run.
    corpus: dict[str, str] = field(default_factory=dict)
    usage: RunUsage = field(default_factory=RunUsage)
    brief: ResearchBrief | None = None
    error: str | None = None
    trace_id: str | None = None
    elapsed_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "trace_id": self.trace_id,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "llm_calls": self.llm_calls,
            "llm_retries": self.llm_attempts - self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_retries": max(0, self.tool_attempts - self.tool_calls),
            "repairs": self.repairs,
            "models_used": self.models_used,
            "fell_back": self.fell_back,
            "sources_retrieved": len(self.corpus),
            "usage": self.usage.to_dict(),
        }


def _final_text(response: Any) -> str:
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def _input_summary(tool_input: dict[str, Any]) -> str:
    return str(next(iter(tool_input.values()))) if tool_input else ""


class ResearchAgent:
    """The agentic loop, with guards.

    - max_iterations bounds the research turns (LLM calls that are not repairs)
    - max_repair_attempts bounds how often a failed brief is sent back to be fixed
    - run_timeout_seconds bounds wall-clock time across every LLM and tool call
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: ToolExecutor | None = None,
        *,
        max_iterations: int | None = None,
        max_repair_attempts: int | None = None,
        run_timeout_seconds: float | None = None,
    ):
        self.llm = llm or LLMClient()
        self.tools = tools or ToolExecutor()
        self.max_iterations = max_iterations or settings.max_iterations
        self.max_repair_attempts = (
            settings.max_repair_attempts if max_repair_attempts is None else max_repair_attempts
        )
        self.run_timeout_seconds = run_timeout_seconds or settings.run_timeout_seconds
        self.state: RunState | None = None

    async def run(self, job_url: str) -> AsyncIterator[AgentStep]:
        """Yields AgentSteps as the run progresses. Never raises for an
        expected failure; the run ends with "complete" or "error", then a
        "run_summary" step carrying usage, cost and the trace id."""
        self.state = state = RunState(job_url=job_url)
        deadline = Deadline(self.run_timeout_seconds)
        started = time.monotonic()

        # Spans are started without attaching them to the current context:
        # an async generator can be resumed in a different context after a
        # yield, and a context token attached before the yield would then fail
        # to detach. trace.use_span() is attached only around awaits that do
        # not yield.
        root = get_tracer().start_span(
            "invoke_agent research-agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "research-agent",
                "gen_ai.request.model": self.llm.primary_model,
                "agent.job_url": job_url,
                "agent.max_iterations": self.max_iterations,
                "agent.run_timeout_s": self.run_timeout_seconds,
            },
        )
        state.trace_id = trace.format_trace_id(root.get_span_context().trace_id)
        try:
            try:
                async for step in self._loop(state, deadline, root):
                    yield step
            except RunTimeoutError:
                yield self._fail(state, f"Run timed out after {self.run_timeout_seconds:.0f}s")
            except (anthropic.APIError, TimeoutError) as e:
                if deadline.expired():
                    yield self._fail(state, f"Run timed out after {self.run_timeout_seconds:.0f}s")
                else:
                    yield self._fail(
                        state, f"LLM call failed after retries: {type(e).__name__}: {e}"
                    )
            except Exception as e:
                log.exception("Agent run failed")
                root.record_exception(e)
                yield self._fail(state, f"Agent run failed: {type(e).__name__}: {e}")

            state.elapsed_seconds = time.monotonic() - started
            summary = state.summary()
            log.info("Run finished: %s", json.dumps(summary))
            yield AgentStep(
                type="run_summary",
                message=(
                    f"{state.status}: {state.llm_calls} LLM calls, {state.tool_calls} tool calls, "
                    f"${state.usage.cost_usd:.4f}"
                ),
                data=summary,
            )
        finally:
            state.elapsed_seconds = time.monotonic() - started
            self._end_root(root, state)

    def _end_root(self, root: Span, state: RunState) -> None:
        if state.status == "running":
            state.status = "cancelled"  # the consumer stopped reading mid-run
        root.set_attribute("agent.status", state.status)
        root.set_attribute("agent.llm_calls", state.llm_calls)
        root.set_attribute("agent.llm_retries", state.llm_attempts - state.llm_calls)
        root.set_attribute("agent.tool_calls", state.tool_calls)
        root.set_attribute("agent.repairs", state.repairs)
        root.set_attribute("agent.fell_back", state.fell_back)
        root.set_attribute("agent.models_used", state.models_used)
        root.set_attribute("agent.sources_retrieved", len(state.corpus))
        root.set_attribute("gen_ai.usage.input_tokens", state.usage.input_tokens)
        root.set_attribute("gen_ai.usage.output_tokens", state.usage.output_tokens)
        root.set_attribute("agent.cost_usd", round(state.usage.cost_usd, 6))
        for model, usage in state.usage.by_model.items():
            root.set_attribute(f"agent.cost_usd.{model}", round(usage.cost_usd, 6))
        if state.status == "error":
            root.set_status(Status(StatusCode.ERROR, state.error or ""))
        root.end()

    def _fail(self, state: RunState, message: str, output: str = "") -> AgentStep:
        state.status = "error"
        state.error = message
        return AgentStep(type="error", message=message, output=output)

    async def _loop(
        self, state: RunState, deadline: Deadline, root: Span
    ) -> AsyncIterator[AgentStep]:
        tracer = get_tracer()
        root_ctx = trace.set_span_in_context(root)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Research this job posting and produce a research brief: {state.job_url}",
            }
        ]

        repairs_before = state.repairs
        while state.llm_calls < self.max_iterations + state.repairs:
            repairing = state.repairs > repairs_before
            repairs_before = state.repairs
            step_span = tracer.start_span(
                "agent.step",
                context=root_ctx,
                attributes={
                    "agent.iteration": state.llm_calls + 1,
                    "agent.step.kind": "repair" if repairing else "research",
                },
            )
            try:
                async for step in self._turn(state, deadline, messages, step_span):
                    yield step
            except Exception as e:
                step_span.record_exception(e)
                step_span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                step_span.end()
            if state.status != "running":
                return

        yield self._fail(
            state, f"Agent hit max iterations ({self.max_iterations}) without completing"
        )

    async def _turn(
        self,
        state: RunState,
        deadline: Deadline,
        messages: list[dict[str, Any]],
        step_span: Span,
    ) -> AsyncIterator[AgentStep]:
        """One LLM call and whatever it asks for: tools, a repair, or the end.
        Appends to messages when the loop should continue."""
        with trace.use_span(step_span, end_on_exit=False):
            call = await self.llm.create(
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
                deadline=deadline,
                usage=state.usage,
            )
        response = call.response
        state.llm_calls += 1
        state.llm_attempts += call.attempts
        state.fell_back = state.fell_back or call.fell_back
        if call.model not in state.models_used:
            state.models_used.append(call.model)
        step_span.set_attribute("agent.stop_reason", str(response.stop_reason))

        if response.stop_reason == "end_turn":
            async for step in self._finish(state, response, messages, step_span):
                yield step
            return

        if response.stop_reason == "tool_use":
            async for step in self._run_tools(state, deadline, response, messages, step_span):
                yield step
            return

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            yield self._fail(
                state,
                f"Model declined the request (refusal{f': {category}' if category else ''})",
            )
            return

        yield self._fail(state, f"Unexpected stop reason: {response.stop_reason}")

    async def _finish(
        self,
        state: RunState,
        response: Any,
        messages: list[dict[str, Any]],
        step_span: Span,
    ) -> AsyncIterator[AgentStep]:
        final_text = _final_text(response)
        with get_tracer().start_as_current_span(
            "agent.validate_brief", context=trace.set_span_in_context(step_span)
        ) as span:
            try:
                brief = parse_brief(final_text, retrieved_urls=set(state.corpus))
                error = None
                span.set_attribute("agent.brief_valid", True)
                span.set_attribute("agent.claims", len(brief.claims))
            except BriefValidationError as e:
                brief, error = None, e
                span.set_attribute("agent.brief_valid", False)
                span.set_attribute("agent.validation_problems", e.problems[:10])

        if error is None:
            state.brief = brief
            state.status = "complete"
            yield AgentStep(
                type="complete", output=brief.model_dump_json(), message="Research complete"
            )
            return

        if state.repairs >= self.max_repair_attempts:
            yield self._fail(
                state,
                f"Failed to parse a valid brief after {state.repairs} repair attempts: {error}",
                output=final_text,
            )
            return

        state.repairs += 1
        yield AgentStep(
            type="repair",
            message=(
                "Brief failed validation, asking for a fix "
                f"({state.repairs}/{self.max_repair_attempts})"
            ),
            output=str(error),
        )
        messages.append({"role": "assistant", "content": response.content})
        problems = "\n".join(f"- {p}" for p in error.problems)
        messages.append({"role": "user", "content": REPAIR_PROMPT.format(problems=problems)})

    async def _run_tools(
        self,
        state: RunState,
        deadline: Deadline,
        response: Any,
        messages: list[dict[str, Any]],
        step_span: Span,
    ) -> AsyncIterator[AgentStep]:
        messages.append({"role": "assistant", "content": response.content})
        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        for block in tool_blocks:
            yield AgentStep(
                type="tool_call",
                tool=block.name,
                input=json.dumps(block.input),
                message=f"Calling {block.name}: {_input_summary(block.input)}",
            )

        # Independent calls in one turn run concurrently; each gets its own
        # execute_tool span under this step. execute() only raises when the
        # run is out of time; wait for every call before re-raising so no
        # sibling task is left running.
        with trace.use_span(step_span, end_on_exit=False):
            outcomes = await asyncio.gather(
                *(self.tools.execute(b.name, b.input, deadline, call_id=b.id) for b in tool_blocks),
                return_exceptions=True,
            )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        results: list[ToolResult] = outcomes

        tool_results = []
        for block, result in zip(tool_blocks, results, strict=True):
            state.tool_calls += 1
            state.tool_attempts += result.attempts
            state.corpus.update(result.sources)

            content = result.content
            display = content[:300] + "..." if len(content) > 300 else content
            status = "failed" if result.is_error else f"returned {len(content)} characters"
            note = f" (fell back to {result.fallback})" if result.fallback else ""
            yield AgentStep(
                type="tool_result",
                tool=block.name,
                output=display,
                message=f"{block.name} {status}{note}",
            )
            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            }
            if result.is_error:
                tool_result["is_error"] = True
            tool_results.append(tool_result)

        # All results go back in one user message so Claude keeps making
        # parallel calls.
        messages.append({"role": "user", "content": tool_results})


async def run_agent(job_url: str, agent: ResearchAgent | None = None) -> AsyncIterator[str]:
    """Run the agent and yield newline-delimited JSON steps (the API's wire format)."""
    agent = agent or ResearchAgent()
    async for step in agent.run(job_url):
        yield stream_step(step)
