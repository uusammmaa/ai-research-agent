"""Fakes for driving agent.orchestrator without hitting the real Anthropic API."""


class FakeBlock:
    """Mimics an Anthropic content block (text or tool_use)."""

    def __init__(self, type_, text=None, name=None, input=None, id=None):
        self.type = type_
        if text is not None:
            self.text = text
        self.name = name
        self.input = input
        self.id = id


class FakeResponse:
    """Mimics an Anthropic Message response."""

    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


def text_response(text: str) -> FakeResponse:
    return FakeResponse(stop_reason="end_turn", content=[FakeBlock("text", text=text)])


def tool_use_response(tool_calls: list[tuple[str, dict]]) -> FakeResponse:
    """tool_calls: list of (tool_name, tool_input) pairs."""
    blocks = []
    for i, (name, tool_input) in enumerate(tool_calls):
        blocks.append(FakeBlock("tool_use", name=name, input=tool_input, id=f"tool_{name}_{i}"))
    return FakeResponse(stop_reason="tool_use", content=blocks)


def make_create_stub(responses: list[FakeResponse]):
    """Returns a callable usable as side_effect for client.messages.create."""
    iterator = iter(responses)

    def _create(*args, **kwargs):
        try:
            return next(iterator)
        except StopIteration:
            # If the loop asks for more turns than we scripted, keep looping
            # on tool_use forever so max_iterations behavior can be exercised.
            # Return a fresh object each time — orchestrator.py appends
            # response.content straight into the shared `messages` list, and
            # returning the same instance repeatedly would alias that list
            # across multiple message-history entries.
            last = responses[-1]
            return FakeResponse(stop_reason=last.stop_reason, content=list(last.content))

    return _create


def make_execute_tool_stub(canned: dict):
    """canned: {tool_name: result_string} or {tool_name: callable(tool_input) -> str}."""

    async def _execute_tool(tool_name, tool_input):
        value = canned.get(tool_name, f"No canned result for {tool_name}")
        if callable(value):
            return value(tool_input)
        return value

    return _execute_tool


VALID_BRIEF = {
    "role": "Senior Backend Engineer",
    "company": "Acme Corp",
    "location": "Remote",
    "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
    "key_requirements": ["5+ years Python", "Distributed systems experience", "API design"],
    "company_summary": "Acme Corp is a Series B fintech startup founded in 2019, building payment infrastructure for SMBs.",
    "culture_signals": ["Async-first, no mandatory meetings", "Small team (30 people)"],
    "talking_points": ["Highlight FastAPI experience", "Mention distributed systems project"],
    "red_flags": [],
    "sources": ["https://jobs.acme.com/senior-backend-engineer", "https://acme.com/about"],
}
