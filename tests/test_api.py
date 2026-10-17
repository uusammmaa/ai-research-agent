import json

import pytest
from fastapi.testclient import TestClient

import main
from evals.fakes import CANNED_TOOLS, VALID_BRIEF, make_agent, research_turns, text_response


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("job_url", ["", "file:///etc/passwd", "not a url"])
def test_run_rejects_non_http_urls(client, job_url):
    response = client.post("/agent/run", json={"job_url": job_url})
    assert response.status_code == 422


def test_run_requires_api_keys(client, monkeypatch):
    monkeypatch.setattr(main.settings, "anthropic_api_key", "")
    response = client.post("/agent/run", json={"job_url": "https://example.com/job/123"})
    assert response.status_code == 503


def test_run_streams_ndjson_steps(client, monkeypatch):
    monkeypatch.setattr(main.settings, "anthropic_api_key", "test")
    monkeypatch.setattr(main.settings, "tavily_api_key", "test")
    agent, _ = make_agent([*research_turns(), text_response(json.dumps(VALID_BRIEF))], CANNED_TOOLS)
    monkeypatch.setattr(main, "build_agent", lambda: agent)

    response = client.post("/agent/run", json={"job_url": "https://example.com/job/123"})
    assert response.status_code == 200
    steps = [json.loads(line) for line in response.text.splitlines()]
    assert [s["type"] for s in steps][-2:] == ["complete", "run_summary"]
    assert steps[-1]["data"]["usage"]["cost_usd"] > 0
