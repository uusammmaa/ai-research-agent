"""Retry, backoff and deadline primitives shared by the LLM and tool layers."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

Sleep = Callable[[float], Awaitable[None]]
OnRetry = Callable[[int, float, BaseException], None]


class RunTimeoutError(Exception):
    """Raised when the run-level deadline is exhausted."""


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    attempt n (1-based) waits uniform(0, min(max_delay, base_delay * 2**(n-1)))
    before attempt n+1. Full jitter avoids synchronised retries when several
    runs hit the same rate limit at once.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return random.uniform(0, ceiling) if self.jitter else ceiling


class Deadline:
    """A monotonic wall-clock budget shared by every call in one run."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.seconds = seconds
        self.expires_at = clock() + seconds

    def remaining(self) -> float:
        return max(0.0, self.expires_at - self._clock())

    def expired(self) -> bool:
        return self.remaining() <= 0

    def cap(self, timeout: float) -> float:
        """Shrink a per-call timeout so it never outlives the run."""
        remaining = self.remaining()
        if remaining <= 0:
            raise RunTimeoutError(f"Run exceeded its {self.seconds:.0f}s time budget")
        return min(timeout, remaining)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[BaseException], bool],
    deadline: Deadline | None = None,
    on_retry: OnRetry | None = None,
    sleep: Sleep = asyncio.sleep,
) -> T:
    """Call fn until it succeeds, a non-retryable error occurs, attempts run
    out, or the next backoff would overrun the deadline. Re-raises the last
    error unchanged so callers can still branch on its type."""
    attempt = 1
    while True:
        try:
            return await fn()
        except Exception as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay_for(attempt)
            if deadline is not None and delay >= deadline.remaining():
                raise
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            await sleep(delay)
            attempt += 1
