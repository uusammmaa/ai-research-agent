# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.orchestrator import run_agent
from config import get_cors_origins

app = FastAPI(title="AI Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


class AgentRequest(BaseModel):
    job_url: str


@app.post("/agent/run")
async def run_agent_endpoint(request: AgentRequest):
    """Run the research agent and stream steps as newline-delimited JSON."""
    return StreamingResponse(run_agent(request.job_url), media_type="text/plain")
