import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anthropic

from agent.llm import LLMClient
from agent.models import AgentStep, ResearchBrief
from agent.output import BriefValidationError, parse_brief
from agent.prompts import REPAIR_PROMPT, SYSTEM_PROMPT
from agent.reliability import Deadline, RunTimeoutError
from agent.tools import TOOL_SCHEMAS, ToolExecutor, ToolResult
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
    brief: ResearchBrief | None = None
    error: str | None = None


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
        expected failure; the last step is either "complete" or "error"."""
        self.state = state = RunState(job_url=job_url)
        deadline = Deadline(self.run_timeout_seconds)
        try:
            async for step in self._loop(state, deadline):
                yield step
        except RunTimeoutError:
            yield self._fail(state, f"Run timed out after {self.run_timeout_seconds:.0f}s")
        except (anthropic.APIError, TimeoutError) as e:
            if deadline.expired():
                yield self._fail(state, f"Run timed out after {self.run_timeout_seconds:.0f}s")
            else:
                yield self._fail(state, f"LLM call failed after retries: {type(e).__name__}: {e}")
        except Exception as e:
            log.exception("Agent run failed")
            yield self._fail(state, f"Agent run failed: {type(e).__name__}: {e}")

    def _fail(self, state: RunState, message: str, output: str = "") -> AgentStep:
        state.status = "error"
        state.error = message
        return AgentStep(type="error", message=message, output=output)

    async def _loop(self, state: RunState, deadline: Deadline) -> AsyncIterator[AgentStep]:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Research this job posting and produce a research brief: {state.job_url}",
            }
        ]

        while state.llm_calls < self.max_iterations + state.repairs:
            call = await self.llm.create(
                system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS, messages=messages, deadline=deadline
            )
            response = call.response
            state.llm_calls += 1
            state.llm_attempts += call.attempts
            state.fell_back = state.fell_back or call.fell_back
            if call.model not in state.models_used:
                state.models_used.append(call.model)

            if response.stop_reason == "end_turn":
                final_text = _final_text(response)
                try:
                    brief = parse_brief(final_text, retrieved_urls=set(state.corpus))
                except BriefValidationError as e:
                    if state.repairs >= self.max_repair_attempts:
                        yield self._fail(
                            state,
                            f"Failed to parse a valid brief after {state.repairs} repair "
                            f"attempts: {e}",
                            output=final_text,
                        )
                        return
                    state.repairs += 1
                    yield AgentStep(
                        type="repair",
                        message=f"Brief failed validation, asking for a fix ({state.repairs}/{self.max_repair_attempts})",
                        output=str(e),
                    )
                    messages.append({"role": "assistant", "content": response.content})
                    problems = "\n".join(f"- {p}" for p in e.problems)
                    messages.append(
                        {"role": "user", "content": REPAIR_PROMPT.format(problems=problems)}
                    )
                    continue

                state.brief = brief
                state.status = "complete"
                yield AgentStep(
                    type="complete", output=brief.model_dump_json(), message="Research complete"
                )
                return

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_blocks = [b for b in response.content if b.type == "tool_use"]

                for block in tool_blocks:
                    yield AgentStep(
                        type="tool_call",
                        tool=block.name,
                        input=json.dumps(block.input),
                        message=f"Calling {block.name}: {_input_summary(block.input)}",
                    )

                # Independent calls in one turn run concurrently.
                results: list[ToolResult] = await asyncio.gather(
                    *(self.tools.execute(b.name, b.input, deadline) for b in tool_blocks)
                )

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

                # All results go back in one user message so Claude keeps
                # making parallel calls.
                messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None) if details else None
                yield self._fail(
                    state,
                    f"Model declined the request (refusal{f': {category}' if category else ''})",
                )
                return

            yield self._fail(state, f"Unexpected stop reason: {response.stop_reason}")
            return

        yield self._fail(
            state, f"Agent hit max iterations ({self.max_iterations}) without completing"
        )


async def run_agent(job_url: str, agent: ResearchAgent | None = None) -> AsyncIterator[str]:
    """Run the agent and yield newline-delimited JSON steps (the API's wire format)."""
    agent = agent or ResearchAgent()
    async for step in agent.run(job_url):
        yield stream_step(step)
