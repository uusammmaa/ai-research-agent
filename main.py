from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.orchestrator import ResearchAgent, run_agent
from config import get_cors_origins, settings

app = FastAPI(title="AI Research Agent API")

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


@app.post("/agent/run")
async def run_agent_endpoint(request: AgentRequest):
    """Run the research agent and stream steps as newline-delimited JSON."""
    if not settings.anthropic_api_key or not settings.tavily_api_key:
        raise HTTPException(
            status_code=503, detail="ANTHROPIC_API_KEY and TAVILY_API_KEY must be configured"
        )
    return StreamingResponse(run_agent(request.job_url, build_agent()), media_type="text/plain")
