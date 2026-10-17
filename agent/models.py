from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://")) and len(value) > len("https://")


class Claim(BaseModel):
    """One factual statement in the brief, tied to the source that supports it."""

    statement: str = Field(min_length=1)
    source: str = Field(description="URL of a page fetched or returned by search in this run")
    quote: str = Field(min_length=1, description="Verbatim excerpt from the source")

    @field_validator("statement", "quote")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("source")
    @classmethod
    def _url(cls, v: str) -> str:
        v = v.strip()
        if not _is_http_url(v):
            raise ValueError(f"must be an http(s) URL, got {v!r}")
        return v


class ResearchBrief(BaseModel):
    role: str = Field(min_length=1)
    company: str = Field(min_length=1)
    location: str
    tech_stack: list[str]
    key_requirements: list[str]
    company_summary: str
    culture_signals: list[str]
    talking_points: list[str]
    red_flags: list[str]
    sources: list[str]
    claims: list[Claim] = Field(min_length=1)

    @field_validator("role", "company")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("sources")
    @classmethod
    def _sources_are_urls(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v]
        bad = [s for s in cleaned if not _is_http_url(s)]
        if bad:
            raise ValueError(f"every source must be an http(s) URL; invalid: {bad}")
        if not cleaned:
            raise ValueError("must list at least one source URL")
        return cleaned

    @model_validator(mode="after")
    def _claims_cite_listed_sources(self) -> "ResearchBrief":
        listed = set(self.sources)
        missing = sorted({c.source for c in self.claims if c.source not in listed})
        if missing:
            raise ValueError(f"claims cite sources not listed in 'sources': {missing}")
        return self


class AgentStep(BaseModel):
    # "tool_call" | "tool_result" | "repair" | "complete" | "error" | "run_summary"
    type: str
    tool: str = ""  # tool name if type == "tool_call"
    input: str = ""  # tool input summary
    output: str = ""  # tool result summary or final brief JSON
    message: str = ""  # human-readable step description
    data: dict[str, Any] | None = None  # structured extras (usage, cost, trace id)
