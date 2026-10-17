"""Model catalog: per-token prices and the request features each model accepts.

Prices are USD per million tokens, Anthropic first-party API rates as of
2026-10. Check https://www.anthropic.com/pricing before relying on them for
billing; they are here to make per-run cost visible, not to reconcile invoices.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    input: float
    output: float
    cache_read: float
    # 5-minute cache writes bill at 1.25x the input rate.
    cache_write: float | None = None
    # Some models charge more once a prompt passes a size threshold.
    long_context_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None
    supports_effort: bool = True
    supports_refusal_fallback: bool = False

    @property
    def cache_write_rate(self) -> float:
        return self.cache_write if self.cache_write is not None else self.input * 1.25


MODELS: dict[str, ModelInfo] = {
    "claude-fable-5-1": ModelInfo(10.00, 50.00, 0.25, supports_refusal_fallback=True),
    "claude-fable-5": ModelInfo(10.00, 50.00, 1.00),
    "claude-opus-5-5": ModelInfo(4.00, 20.00, 0.20, supports_refusal_fallback=True),
    "claude-opus-5": ModelInfo(5.00, 25.00, 0.50, supports_refusal_fallback=True),
    "claude-opus-4-8": ModelInfo(5.00, 25.00, 0.50),
    "claude-opus-4-7": ModelInfo(5.00, 25.00, 0.50),
    "claude-opus-4-6": ModelInfo(5.00, 25.00, 0.50),
    "claude-sonnet-5-5": ModelInfo(2.00, 10.00, 0.20, supports_refusal_fallback=True),
    "claude-sonnet-5": ModelInfo(2.00, 10.00, 0.20),
    "claude-sonnet-4-6": ModelInfo(3.00, 15.00, 0.30),
    "claude-haiku-5-5": ModelInfo(
        0.10,
        0.50,
        0.01,
        long_context_threshold=100_000,
        long_input=0.50,
        long_output=2.50,
    ),
    "claude-haiku-4-5": ModelInfo(1.00, 5.00, 0.10, supports_effort=False),
}


def model_info(model: str) -> ModelInfo | None:
    return MODELS.get(model)


def call_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """USD cost of one API call, or None for a model not in the catalog."""
    info = MODELS.get(model)
    if info is None:
        return None
    input_rate, output_rate = info.input, info.output
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    if info.long_context_threshold is not None and prompt_tokens > info.long_context_threshold:
        input_rate = info.long_input or input_rate
        output_rate = info.long_output or output_rate
    total = (
        input_tokens * input_rate
        + output_tokens * output_rate
        + cache_read_tokens * info.cache_read
        + cache_write_tokens * info.cache_write_rate
    )
    return total / 1_000_000
