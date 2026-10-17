# AI Research Agent

A Python research agent that takes a question, plans a search strategy, gathers sources and returns a synthesised answer with citations.

**Status: in progress.** The agent backend and tool layer are in place. Retrieval, synthesis and the evaluation harness are being built next.

## Why this exists

Most research-agent demos wire a model to a search API and stop there. The interesting problems start after that: deciding when the agent has gathered enough, keeping it from looping, recovering when a tool call fails, and knowing whether the answer is actually grounded in the sources it cites. This repo is where I work those out in the open.

## Design

**Tool layer.** Each capability the agent can call lives in agent/tools.py behind a single interface, so adding a capability means adding one function rather than touching the control loop.

**Configuration.** Model, keys and runtime settings are read from environment variables through config.py. Copy .env.example to .env and fill it in. Nothing is committed.

**Control loop.** The agent plans, calls tools, evaluates what it has, and decides whether to continue or answer. Step limits and a stop condition prevent runaway loops.

## Running it

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in your API key in .env, then run the agent from the agent package.

## Roadmap

- Source retrieval and ranking
- Answer synthesis with inline citations
- Evaluation harness: groundedness and citation accuracy scored against a fixed question set
- Tracing and per-run token cost
- Retry and fallback on tool failure
