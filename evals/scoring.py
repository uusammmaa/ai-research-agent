"""Shared scoring logic for the mocked and live eval suites."""

from pydantic import ValidationError

from agent.models import ResearchBrief


def is_valid_research_brief(brief: dict) -> bool:
    """True if the brief validates against the same pydantic model the agent
    uses, so the eval and the agent cannot drift apart."""
    try:
        ResearchBrief.model_validate(brief)
        return True
    except (ValidationError, TypeError):
        return False
