"""Shared scoring logic for the mocked and live eval suites."""

REQUIRED_BRIEF_FIELDS = {
    "role",
    "company",
    "location",
    "tech_stack",
    "key_requirements",
    "company_summary",
    "culture_signals",
    "talking_points",
    "red_flags",
    "sources",
}


def is_valid_research_brief(brief: dict) -> bool:
    """Mirrors the ResearchBrief schema in agent/models.py: required fields
    present, list-typed fields are lists, and the identifying fields are non-empty."""
    try:
        return (
            REQUIRED_BRIEF_FIELDS.issubset(brief.keys())
            and isinstance(brief["tech_stack"], list)
            and isinstance(brief["sources"], list)
            and bool(brief["role"])
            and bool(brief["company"])
        )
    except (KeyError, TypeError):
        return False
