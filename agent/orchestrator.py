# backend/agent/orchestrator.py
import json
import re
from collections.abc import AsyncIterator

import anthropic

from agent.models import AgentStep, ResearchBrief
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOL_SCHEMAS, execute_tool
from config import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def stream_step(step: AgentStep) -> str:
    """Serialise an AgentStep as a newline-delimited JSON string for SSE."""
    return json.dumps(step.model_dump()) + "\n"


def _extract_json(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate

    start = 0
    while True:
        idx = text.find("{", start)
        if idx == -1:
            break
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, idx)
            if isinstance(obj, dict):
                return json.dumps(obj)
        except json.JSONDecodeError:
            pass
        start = idx + 1

    return text


async def run_agent(job_url: str) -> AsyncIterator[str]:
    """
    The agentic loop.

    Yields serialised AgentStep objects as newline-delimited JSON.
    The frontend reads these and renders each step in real time.
    """

    messages = [
        {
            "role": "user",
            "content": f"Research this job posting and produce a research brief: {job_url}",
        }
    ]

    iteration = 0

    while iteration < settings.max_iterations:
        iteration += 1

        response = client.messages.create(
            model=settings.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            try:
                brief_data = json.loads(_extract_json(final_text))
                brief = ResearchBrief(**brief_data)
                yield stream_step(
                    AgentStep(
                        type="complete", output=brief.model_dump_json(), message="Research complete"
                    )
                )
            except Exception as e:
                yield stream_step(
                    AgentStep(
                        type="error",
                        message=f"Failed to parse final brief: {str(e)}",
                        output=final_text,
                    )
                )
            return

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input

                input_summary = str(list(tool_input.values())[0]) if tool_input else ""
                yield stream_step(
                    AgentStep(
                        type="tool_call",
                        tool=tool_name,
                        input=json.dumps(tool_input),
                        message=f"Calling {tool_name}: {input_summary}",
                    )
                )

                result = await execute_tool(tool_name, tool_input)

                display_result = result[:300] + "..." if len(result) > 300 else result
                yield stream_step(
                    AgentStep(
                        type="tool_result",
                        tool=tool_name,
                        output=display_result,
                        message=f"{tool_name} returned {len(result)} characters",
                    )
                )

                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

            messages.append({"role": "user", "content": tool_results})

        else:
            yield stream_step(
                AgentStep(type="error", message=f"Unexpected stop reason: {response.stop_reason}")
            )
            return

    yield stream_step(
        AgentStep(
            type="error",
            message=f"Agent hit max iterations ({settings.max_iterations}) without completing",
        )
    )
