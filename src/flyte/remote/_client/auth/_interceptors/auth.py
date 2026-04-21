from __future__ import annotations

import typing

from connectrpc.code import Code
from connectrpc.errors import ConnectError

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


class AuthUnaryInterceptor(_BaseAuthInterceptor):
    """ConnectRPC unary interceptor that injects auth headers and retries on UNAUTHENTICATED."""

    async def intercept_unary(self, call_next, request, ctx):
        auth_headers = await self._inject_auth_headers(ctx)
        try:
            return await call_next(request, ctx)
        except ConnectError as e:
            if e.code in _RETRYABLE_AUTH_CODES:
                await self._refresh_and_reinject(auth_headers, ctx)
                return await call_next(request, ctx)
            raise


class AuthClientStreamInterceptor(_BaseAuthInterceptor):
    """ConnectRPC client-stream interceptor that injects auth headers and retries on UNAUTHENTICATED.

    NOTE: On retry, the same ``request`` async iterator is passed to ``call_next``
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
            if e.code in _RETRYABLE_AUTH_CODES:
                await self._refresh_and_reinject(auth_headers, ctx)
                return await call_next(request, ctx)
            raise


class AuthServerStreamInterceptor(_BaseAuthInterceptor):
    """ConnectRPC server-stream interceptor that injects auth headers and retries on UNAUTHENTICATED."""

    async def intercept_server_stream(self, call_next, request, ctx):
        auth_headers = await self._inject_auth_headers(ctx)
        try:
            async for response in call_next(request, ctx):
                yield response
        except ConnectError as e:
            if e.code in _RETRYABLE_AUTH_CODES:
                await self._refresh_and_reinject(auth_headers, ctx)
                async for response in call_next(request, ctx):
                    yield response
            else:
                raise


class AuthBidiStreamInterceptor(_BaseAuthInterceptor):
    """ConnectRPC bidi-stream interceptor that injects auth headers and retries on UNAUTHENTICATED.

    See AuthClientStreamInterceptor for the request-iterator replay caveat.
    """

    async def intercept_bidi_stream(self, call_next, request, ctx):
        auth_headers = await self._inject_auth_headers(ctx)
        try:
            async for response in call_next(request, ctx):
                yield response
        except ConnectError as e:
            if e.code in _RETRYABLE_AUTH_CODES:
                await self._refresh_and_reinject(auth_headers, ctx)
                async for response in call_next(request, ctx):
                    yield response
            else:
                raise
