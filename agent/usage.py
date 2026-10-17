"""Per-run token usage and cost, broken down by model."""

from dataclasses import asdict, dataclass, field
from typing import Any

from agent.pricing import call_cost


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    # False if any call ran on a model missing from the price catalog.
    cost_known: bool = True


@dataclass
class CallUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None


def read_usage(usage: Any) -> tuple[int, int, int, int]:
    """(input, output, cache_read, cache_write) from an SDK Usage object."""
    if usage is None:
        return 0, 0, 0, 0
    return (
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


@dataclass
class RunUsage:
    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    def record(self, model: str, usage: Any) -> CallUsage:
        inp, out, cread, cwrite = read_usage(usage)
        cost = call_cost(
            model,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cread,
            cache_write_tokens=cwrite,
        )
        m = self.by_model.setdefault(model, ModelUsage())
        m.calls += 1
        m.input_tokens += inp
        m.output_tokens += out
        m.cache_read_tokens += cread
        m.cache_write_tokens += cwrite
        if cost is None:
            m.cost_known = False
        else:
            m.cost_usd += cost
        return CallUsage(inp, out, cread, cwrite, cost)

    @property
    def input_tokens(self) -> int:
        return sum(
            m.input_tokens + m.cache_read_tokens + m.cache_write_tokens
            for m in self.by_model.values()
        )

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.by_model.values())

    @property
    def cost_usd(self) -> float:
        return sum(m.cost_usd for m in self.by_model.values())

    @property
    def cost_known(self) -> bool:
        return all(m.cost_known for m in self.by_model.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_known": self.cost_known,
            "by_model": {
                model: {**asdict(m), "cost_usd": round(m.cost_usd, 6)}
                for model, m in self.by_model.items()
            },
        }
