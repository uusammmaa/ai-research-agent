"""Groundedness and citation-accuracy scoring for research briefs.

Two questions per claim:

1. Citation accuracy: does the claim cite a source that a tool call in this
   run actually retrieved? (A URL the agent never read is a fabricated
   citation, however plausible it looks.)
2. Groundedness: does the cited source support the claim? Checked
   deterministically, so it runs in CI with no model calls:
     - the claim's quote appears verbatim (after whitespace, case and
       punctuation normalisation) in the cited source's retrieved text,
     - every number in the statement appears in that source text, and
     - most of the statement's content words appear in that source text.

The deterministic check is strict about provenance and lenient about
paraphrase. It catches fabricated quotes, misattributed sources and changed
numbers; it can miss a statement that reuses the source's words to say
something the source does not. evals/judge.py adds an optional LLM judge for
that case in the live suite.
"""

import re
import unicodedata
from dataclasses import asdict, dataclass

from agent.output import normalize_url

CONTENT_WORD_THRESHOLD = 0.5

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has",
    "have", "had", "its", "their", "they", "them", "our", "you", "your", "who", "which",
    "will", "would", "can", "could", "into", "about", "over", "than", "then", "also",
    "within", "based", "such", "per", "via", "not", "but", "all", "any", "out",
}  # fmt: skip


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    )
    text = re.sub(r"[^\w\s$%.'-]", " ", text)
    # Keep decimal points ("5.5"), drop sentence punctuation.
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_edges(text: str) -> str:
    return text.strip(" .,;:'\"-")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9'-]{2,}", text) if w not in _STOPWORDS}


@dataclass
class ClaimScore:
    statement: str
    source: str
    cites_retrieved_source: bool
    quote_found: bool
    numbers_supported: bool
    content_overlap: float
    supported: bool
    reason: str


def score_claim(claim: dict, corpus: dict[str, str]) -> ClaimScore:
    statement = str(claim.get("statement", ""))
    source = str(claim.get("source", ""))
    quote = str(claim.get("quote", ""))
    by_url = {normalize_url(u): text for u, text in corpus.items()}
    source_text = by_url.get(normalize_url(source)) if source else None

    if source_text is None:
        return ClaimScore(
            statement, source, False, False, False, 0.0, False, "source not retrieved"
        )

    norm_source = _normalize_text(source_text)
    norm_quote = _strip_edges(_normalize_text(quote))
    norm_statement = _normalize_text(statement)

    quote_found = bool(norm_quote) and norm_quote in norm_source
    source_numbers = _numbers(norm_source)
    numbers_supported = _numbers(norm_statement) <= source_numbers
    words = _content_words(norm_statement)
    overlap = len(words & _content_words(norm_source)) / len(words) if words else 1.0

    reasons = []
    if not quote_found:
        reasons.append("quote not in cited source")
    if not numbers_supported:
        missing = sorted(_numbers(norm_statement) - source_numbers)
        reasons.append(f"numbers not in source: {missing}")
    if overlap < CONTENT_WORD_THRESHOLD:
        reasons.append(f"low content overlap ({overlap:.2f})")
    supported = not reasons
    return ClaimScore(
        statement,
        source,
        True,
        quote_found,
        numbers_supported,
        round(overlap, 3),
        supported,
        "; ".join(reasons) or "supported",
    )


def grounding_report(brief: dict, corpus: dict[str, str]) -> dict:
    """Scores every claim in a brief against the run's retrieved corpus.

    citation_accuracy: share of claims citing a retrieved source
    groundedness:      share of claims whose cited source supports them
    source_precision:  share of the brief's listed sources that were retrieved
    """
    claims = brief.get("claims") or []
    scores = [score_claim(c, corpus) for c in claims]
    retrieved = {normalize_url(u) for u in corpus}
    sources = brief.get("sources") or []

    def share(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "claims": len(scores),
        "citation_accuracy": share([s.cites_retrieved_source for s in scores]),
        "groundedness": share([s.supported for s in scores]),
        "source_precision": share([normalize_url(s) in retrieved for s in sources]),
        "unsupported": [asdict(s) for s in scores if not s.supported],
    }
