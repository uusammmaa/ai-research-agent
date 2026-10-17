from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from agent.orchestrator import ResearchAgent, run_agent
from agent.telemetry import setup_tracing, shutdown_tracing
from config import get_cors_origins, settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_tracing()
    yield
    shutdown_tracing()  # flush batched spans before the process exits


app = FastAPI(title="AI Research Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_agent() -> ResearchAgent:
    """One agent per request: RunState and the sticky fallback model are per run."""
    return ResearchAgent()


@app.get("/health")
async def health():
    return {"status": "ok"}


class AgentRequest(BaseModel):
    job_url: str

    @field_validator("job_url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")) or len(v) > 2048:
            raise ValueError("job_url must be an http(s) URL")
        return v


@app.post("/agent/run")
async def run_agent_endpoint(request: AgentRequest):
    """Run the research agent and stream steps as newline-delimited JSON.

    The last line is a run_summary step with token usage, cost per model and
    the trace id to look the run up in the tracing backend.
    """
    if not settings.anthropic_api_key or not settings.tavily_api_key:
        raise HTTPException(
            status_code=503, detail="ANTHROPIC_API_KEY and TAVILY_API_KEY must be configured"
        )
    return StreamingResponse(run_agent(request.job_url, build_agent()), media_type="text/plain")
