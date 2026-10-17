from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Keys are optional at import time so tests, evals and CI run without
    # them; the API checks they are present before starting a real run.
    anthropic_api_key: str = ""
    tavily_api_key: str = ""

    # Models
    model: str = "claude-opus-5-5"
    fallback_model: str = "claude-sonnet-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = 16000
    # Server-side refusal fallback (beta) on the models that support it.
    refusal_fallback: bool = True
    # Automatic prompt caching of the repeated system/tools/history prefix.
    prompt_caching: bool = True

    # Run guards
    max_iterations: int = 10
    max_repair_attempts: int = 2
    run_timeout_seconds: float = 300
    llm_timeout_seconds: float = 120
    tool_timeout_seconds: float = 20

    # Retries (attempts include the first try)
    llm_max_attempts: int = 3
    tool_max_attempts: int = 3

    # Tracing: console | console_json | otlp | langfuse | langsmith | none (comma-separated)
    tracing_exporter: str = "console"
    tracing_service_name: str = "ai-research-agent"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "ai-research-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    cors_origins: str = "http://localhost:3000"


settings = Settings()


def get_cors_origins() -> list[str]:
    return [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
