from __future__ import annotations

import typing

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from flyte._logging import logger

if typing.TYPE_CHECKING:
    from flyte.remote._client.auth._authenticators.base import Authenticator, AuthHeaders


class _BaseAuthInterceptor:
    """Base class providing lazy authenticator initialization and header injection."""

    def __init__(self, get_authenticator: typing.Callable[[], Authenticator]):
        self._get_authenticator = get_authenticator
        self._authenticator: Authenticator | None = None

    @property
    def authenticator(self) -> Authenticator:
        if self._authenticator is None:
            self._authenticator = self._get_authenticator()
        return self._authenticator

    async def _inject_auth_headers(self, ctx, *, previous: AuthHeaders | None = None) -> AuthHeaders | None:
        """Inject auth headers into request context, removing any previously injected headers first."""
        # The old gRPC interceptor rebuilt ClientCallDetails from scratch on each attempt, so stale
        # auth headers could never accumulate across retries. ConnectRPC's RequestContext is mutable
        # and shared across retries, so we must explicitly remove headers from the previous attempt
        # before injecting fresh ones — otherwise a header key change (e.g. "authorization" →
        # "flyte-authorization") leaves the stale key behind.
        if previous is not None:
            headers = ctx.request_headers()
            for key in previous.headers:
                headers.pop(key, None)

        auth_headers = await self.authenticator.get_auth_headers()
        if auth_headers:
            ctx.request_headers().update(auth_headers.headers)
        return auth_headers

    async def _refresh_and_reinject(self, previous: AuthHeaders | None, ctx) -> None:
        """Refresh credentials and re-inject auth headers, removing stale ones."""
        await self.authenticator.refresh_credentials(creds_id=previous.creds_id if previous else None)
        await self._inject_auth_headers(ctx, previous=previous)


_RETRYABLE_AUTH_CODES = frozenset({Code.UNAUTHENTICATED, Code.UNKNOWN})

# When a server returns a JSON 401 response whose "code" field is not a valid
# ConnectRPC code string (e.g. "UNAUTHENTICATED" uppercase instead of
# "unauthenticated" lowercase), ConnectWireError.from_dict falls back to
# Code.UNAVAILABLE.  The code-based check above misses that case, so we also
# inspect the error message for common 401-related keywords.
_AUTH_MESSAGE_KEYWORDS = ("unauthorized", "unauthenticated")


def _is_auth_retriable(e: ConnectError, *, had_cached_credentials: bool = True) -> bool:
    """Return True if the error looks like an authentication failure that
    should trigger a credential refresh + retry.

    Args:
        e: The ConnectError that was raised
        had_cached_credentials: Whether the initial request had cached credentials.
            When False (first attempt, no cached credentials), PERMISSION_DENIED is
            treated as retriable since it may be due to missing authentication rather
            than insufficient authorization. When True (had credentials), PERMISSION_DENIED
            is not retriable since it's a legitimate authorization failure.
    """
    if e.code in _RETRYABLE_AUTH_CODES:
        return True

    # For PERMISSION_DENIED (403), only retry if we didn't have cached credentials
    # on the first attempt. This allows PKCE clients to trigger authentication when
    # the server returns 403 (instead of 401) for missing Authorization headers.
    # If we already had credentials, 403 is a legitimate authorization failure.
    if e.code == Code.PERMISSION_DENIED and not had_cached_credentials:
        return True

    msg = e.message.lower()
    return any(kw in msg for kw in _AUTH_MESSAGE_KEYWORDS)


class AuthUnaryInterceptor(_BaseAuthInterceptor):
    """ConnectRPC unary interceptor that injects auth headers and retries on UNAUTHENTICATED.

    Also retries on PERMISSION_DENIED (403) on the first attempt if no cached credentials
    were available, allowing PKCE clients to trigger authentication when the server
    returns 403 instead of 401 for missing Authorization headers.
    """

    async def intercept_unary(self, call_next, request, ctx):
        auth_headers = await self._inject_auth_headers(ctx)
        try:
            return await call_next(request, ctx)
        except ConnectError as e:
            if _is_auth_retriable(e, had_cached_credentials=auth_headers is not None):
                logger.debug("Auth interceptor retrying after %s (code=%s)", e.message, e.code)
                await self._refresh_and_reinject(auth_headers, ctx)
                return await call_next(request, ctx)
            raise


class AuthClientStreamInterceptor(_BaseAuthInterceptor):
    """ConnectRPC client-stream interceptor that injects auth headers and retries on UNAUTHENTICATED.

    Also retries on PERMISSION_DENIED (403) on the first attempt if no cached credentials
    were available, allowing PKCE clients to trigger authentication when the server
    returns 403 instead of 401 for missing Authorization headers.

    NOTE: On retry, the same `request` async iterator is passed to `call_next`
    again. This is only safe when the auth failure occurs before the iterator is
    consumed (the typical case — the server rejects the request headers immediately).
    If the first attempt partially consumes the iterator, the retry will see an
    incomplete stream. This matches the old gRPC AuthStreamUnaryInterceptor behavior.
    """

    async def intercept_client_stream(self, call_next, request, ctx):
        auth_headers = await self._inject_auth_headers(ctx)
        try:
            return await call_next(request, ctx)
        except ConnectError as e:
            if _is_auth_retriable(e, had_cached_credentials=auth_headers is not None):
                logger.debug("Auth interceptor retrying after %s (code=%s)", e.message, e.code)
                await self._refresh_and_reinject(auth_headers, ctx)
                return await call_next(request, ctx)
            raise


class AuthServerStreamInterceptor(_BaseAuthInterceptor):
    """ConnectRPC server-stream interceptor that injects auth headers and retries on UNAUTHENTICATED.

    Also retries on PERMISSION_DENIED (403) on the first attempt if no cached credentials
    were available, allowing PKCE clients to trigger authentication when the server
    returns 403 instead of 401 for missing Authorization headers.
    """

    async def intercept_server_stream(self, call_next, request, ctx):
        auth_headers = await self._inject_auth_headers(ctx)
        try:
            async for response in call_next(request, ctx):
                yield response
        except ConnectError as e:
            if _is_auth_retriable(e, had_cached_credentials=auth_headers is not None):
                logger.debug("Auth interceptor retrying after %s (code=%s)", e.message, e.code)
                await self._refresh_and_reinject(auth_headers, ctx)
                async for response in call_next(request, ctx):
                    yield response
            else:
                raise


class AuthBidiStreamInterceptor(_BaseAuthInterceptor):
    """ConnectRPC bidi-stream interceptor that injects auth headers and retries on UNAUTHENTICATED.

    Also retries on PERMISSION_DENIED (403) on the first attempt if no cached credentials
    were available, allowing PKCE clients to trigger authentication when the server
    returns 403 instead of 401 for missing Authorization headers.

    See AuthClientStreamInterceptor for the request-iterator replay caveat.
    """

    async def intercept_bidi_stream(self, call_next, request, ctx):
        auth_headers = await self._inject_auth_headers(ctx)
        try:
            async for response in call_next(request, ctx):
                yield response
        except ConnectError as e:
            if _is_auth_retriable(e, had_cached_credentials=auth_headers is not None):
                logger.debug("Auth interceptor retrying after %s (code=%s)", e.message, e.code)
                await self._refresh_and_reinject(auth_headers, ctx)
                async for response in call_next(request, ctx):
                    yield response
            else:
                raise
