"""Optional LLM judge for claim support (live suite only, --judge).

The deterministic check in evals/grounding.py verifies provenance: the
quote is really on the cited page. It cannot tell whether the statement
says what the page says. This judge asks a cheaper model exactly that, one
claim at a time, and only for claims that cite a retrieved source, so it
never pays to judge a fabricated citation.
"""

import json
import os

import anthropic

from agent.output import normalize_url
from agent.pricing import call_cost

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-5-5")
MAX_SOURCE_CHARS = 12_000

JUDGE_PROMPT = """You are checking whether a source text supports a claim.

<source url="{url}">
{source}
</source>

<claim>{statement}</claim>

Answer "supported" only if the source text states or directly implies the claim. If the claim adds details, numbers or certainty the source does not have, answer "unsupported"."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def judge_claims(claims: list[dict], corpus: dict[str, str], client=None) -> dict:
    client = client or anthropic.Anthropic()
    by_url = {normalize_url(u): text for u, text in corpus.items()}
    verdicts, cost = [], 0.0
    for claim in claims:
        source_text = by_url.get(normalize_url(str(claim.get("source", ""))))
        if source_text is None:
            verdicts.append(
                {
                    "statement": claim.get("statement"),
                    "verdict": "unsupported",
                    "reason": "source not retrieved",
                }
            )
            continue
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        url=claim["source"],
                        source=source_text[:MAX_SOURCE_CHARS],
                        statement=claim.get("statement", ""),
                    ),
                }
            ],
        )
        cost += (
            call_cost(
                JUDGE_MODEL,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            or 0.0
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            verdict = json.loads(text)
        except json.JSONDecodeError:
            verdict = {
                "verdict": "unsupported",
                "reason": f"unparseable judge output: {text[:200]}",
            }
        verdicts.append({"statement": claim.get("statement"), **verdict})

    supported = [v["verdict"] == "supported" for v in verdicts]
    return {
        "judge_model": JUDGE_MODEL,
        "judge_groundedness": round(sum(supported) / len(supported), 3) if supported else None,
        "judge_cost_usd": round(cost, 6),
        "verdicts": verdicts,
    }
