# backend/agent/models.py

from pydantic import BaseModel


class ResearchBrief(BaseModel):
    role: str
    company: str
    location: str
    tech_stack: list[str]
    key_requirements: list[str]
    company_summary: str
    culture_signals: list[str]
    talking_points: list[str]
    red_flags: list[str]
    sources: list[str]


class AgentStep(BaseModel):
    type: str  # "thinking" | "tool_call" | "tool_result" | "complete" | "error"
    tool: str = ""  # tool name if type == "tool_call"
    input: str = ""  # tool input summary
    output: str = ""  # tool result summary or final brief JSON
    message: str = ""  # human-readable step description
