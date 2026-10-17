"""Claude client wrapper: per-call timeout, retries with backoff, and a
fallback model when the primary stays unavailable."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import anthropic

from agent.pricing import model_info
from agent.reliability import Deadline, RetryPolicy, retry_async
from config import settings

log = logging.getLogger(__name__)

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def is_retryable_llm_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    # APITimeoutError is a subclass of APIConnectionError.
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        # 408 timeout, 409 conflict, 429 rate limit, 5xx including 529 overloaded.
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


@dataclass
class LLMCall:
    response: Any
    model: str
    attempts: int
    fell_back: bool


def default_client() -> anthropic.AsyncAnthropic:
    # SDK retries are off so every retry goes through retry_async, where it
    # is counted and traced, and respects the run deadline.
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None, max_retries=0)


class LLMClient:
    def __init__(
        self,
        client: Any = None,
        *,
        model: str | None = None,
        fallback_model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        refusal_fallback: bool | None = None,
        retry_policy: RetryPolicy | None = None,
        timeout: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._client = client if client is not None else default_client()
        self.primary_model = model or settings.model
        self.fallback_model = (
            fallback_model if fallback_model is not None else settings.fallback_model
        )
        self.effort = effort or settings.effort
        self.max_tokens = max_tokens or settings.max_tokens
        self.refusal_fallback = (
            settings.refusal_fallback if refusal_fallback is None else refusal_fallback
        )
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=settings.llm_max_attempts)
        self.timeout = timeout if timeout is not None else settings.llm_timeout_seconds
        self._sleep = sleep
        # Once the primary is exhausted the rest of the run stays on the
        # fallback rather than paying the retry cost again on every turn.
        self.active_model = self.primary_model

    def request_params(self, model: str) -> dict[str, Any]:
        params: dict[str, Any] = {"model": model, "max_tokens": self.max_tokens}
        info = model_info(model)
        if info is not None and info.supports_effort:
            params["output_config"] = {"effort": self.effort}
        if self.refusal_fallback and info is not None and info.supports_refusal_fallback:
            # If a safety classifier declines, the API re-runs the request on
            # a model chosen for that refusal category, inside the same call.
            params["extra_headers"] = {"anthropic-beta": REFUSAL_FALLBACK_BETA}
            params["extra_body"] = {"fallbacks": "default"}
        return params

    async def _create_with_retries(
        self, model: str, request: dict[str, Any], deadline: Deadline | None, counter: list[int]
    ) -> Any:
        params = {**self.request_params(model), **request}

        async def attempt():
            counter[0] += 1
            timeout = deadline.cap(self.timeout) if deadline else self.timeout
            return await asyncio.wait_for(self._client.messages.create(**params), timeout=timeout)

        def on_retry(attempt_no: int, delay: float, exc: BaseException) -> None:
            log.warning(
                "LLM call to %s failed (attempt %d): %r; retrying in %.2fs",
                model,
                attempt_no,
                exc,
                delay,
            )

        return await retry_async(
            attempt,
            policy=self.retry_policy,
            is_retryable=is_retryable_llm_error,
            deadline=deadline,
            on_retry=on_retry,
            sleep=self._sleep,
        )

    async def create(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        deadline: Deadline | None = None,
    ) -> LLMCall:
        request = {"system": system, "tools": tools, "messages": messages}
        counter = [0]
        model = self.active_model
        try:
            response = await self._create_with_retries(model, request, deadline, counter)
            return LLMCall(response, model, counter[0], fell_back=False)
        except Exception as exc:
            can_fall_back = (
                is_retryable_llm_error(exc)
                and self.fallback_model
                and self.fallback_model != model
                and not (deadline and deadline.expired())
            )
            if not can_fall_back:
                raise
            log.warning(
                "Primary model %s unavailable (%r); falling back to %s",
                model,
                exc,
                self.fallback_model,
            )

        self.active_model = self.fallback_model
        response = await self._create_with_retries(self.active_model, request, deadline, counter)
        return LLMCall(response, self.active_model, counter[0], fell_back=True)
