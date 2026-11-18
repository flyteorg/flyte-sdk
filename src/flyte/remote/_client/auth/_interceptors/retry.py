from __future__ import annotations

import asyncio
import random

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from flyte._logging import logger

from ..errors import AuthenticationTransientError

RETRYABLE_CODES = frozenset({Code.UNAVAILABLE, Code.INTERNAL})


def _is_retryable(e: Exception) -> bool:
    """Whether an error raised through the interceptor chain is transient and worth retrying.

    Besides transient RPC codes, this covers AuthenticationTransientError: the token endpoint answered 5xx/429
    while the auth interceptor (which sits inside this one) was refreshing credentials.
    """
    if isinstance(e, ConnectError):
        return e.code in RETRYABLE_CODES
    return isinstance(e, AuthenticationTransientError)


def _log_retry(ctx, e: Exception, delay: float, attempt: int, max_attempts: int) -> None:
    method = getattr(getattr(ctx, "method", None), "name", "rpc")
    code = e.code if isinstance(e, ConnectError) else type(e).__name__
    message = e.message if isinstance(e, ConnectError) else str(e)
    logger.debug(
        "%s failed (code=%s): %s; retrying in %.1fs (attempt %d/%d)",
        method,
        code,
        message,
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
            except (ConnectError, AuthenticationTransientError) as e:
                if not _is_retryable(e) or attempt == self._max_attempts - 1:
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
            except (ConnectError, AuthenticationTransientError) as e:
                if not _is_retryable(e) or attempt == self._max_attempts - 1:
                    raise
                delay = backoff * (0.5 + random.random())
                _log_retry(ctx, e, delay, attempt, self._max_attempts)
                await asyncio.sleep(delay)
                backoff = min(backoff * self._multiplier, self._max_backoff)
