from __future__ import annotations

import asyncio
import random

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from flyte._logging import logger

RETRYABLE_CODES = frozenset({Code.UNAVAILABLE, Code.RESOURCE_EXHAUSTED, Code.INTERNAL})


def _log_retry(ctx, e: ConnectError, delay: float, attempt: int, max_attempts: int) -> None:
    method = getattr(getattr(ctx, "method", None), "name", "rpc")
    if e.code == Code.RESOURCE_EXHAUSTED:
        # The server explicitly asked us to back off (e.g. a queue at max
        # depth); surface why the call is pausing instead of sleeping silently.
        logger.warning(
            "%s rejected: %s; retrying in %.1fs (attempt %d/%d)", method, e.message, delay, attempt + 1, max_attempts
        )
    else:
        logger.debug(
            "%s failed (code=%s): %s; retrying in %.1fs (attempt %d/%d)",
            method,
            e.code,
            e.message,
            delay,
            attempt + 1,
            max_attempts,
        )


class RetryUnaryInterceptor:
    """ConnectRPC unary interceptor that retries transient failures with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = 4,
        initial_backoff: float = 0.5,
        max_backoff: float = 10.0,
        multiplier: float = 2.0,
    ):
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._multiplier = multiplier

    async def intercept_unary(self, call_next, request, ctx):
        backoff = self._initial_backoff
        for attempt in range(self._max_attempts):
            try:
                return await call_next(request, ctx)
            except ConnectError as e:
                if e.code not in RETRYABLE_CODES or attempt == self._max_attempts - 1:
                    raise
                delay = backoff * (0.5 + random.random())
                _log_retry(ctx, e, delay, attempt, self._max_attempts)
                await asyncio.sleep(delay)
                backoff = min(backoff * self._multiplier, self._max_backoff)


class RetryServerStreamInterceptor:
    """ConnectRPC server-stream interceptor that retries transient failures with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = 4,
        initial_backoff: float = 0.5,
        max_backoff: float = 10.0,
        multiplier: float = 2.0,
    ):
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._multiplier = multiplier

    async def intercept_server_stream(self, call_next, request, ctx):
        backoff = self._initial_backoff
        for attempt in range(self._max_attempts):
            try:
                async for response in call_next(request, ctx):
                    yield response
                return  # Stream completed successfully
            except ConnectError as e:
                if e.code not in RETRYABLE_CODES or attempt == self._max_attempts - 1:
                    raise
                delay = backoff * (0.5 + random.random())
                _log_retry(ctx, e, delay, attempt, self._max_attempts)
                await asyncio.sleep(delay)
                backoff = min(backoff * self._multiplier, self._max_backoff)
