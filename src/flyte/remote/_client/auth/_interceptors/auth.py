from __future__ import annotations

import typing

from connectrpc.code import Code
from connectrpc.errors import ConnectError

if typing.TYPE_CHECKING:
    from flyte.remote._client.auth._authenticators.base import Authenticator


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

    async def _inject_auth_headers(self, ctx) -> str:
        """Inject auth headers into request context. Returns creds_id for refresh tracking."""
        auth_headers = await self.authenticator.get_auth_headers()
        if auth_headers:
            ctx.request_headers().update(auth_headers.headers)
            return auth_headers.creds_id
        return ""

    async def _refresh_and_reinject(self, creds_id: str, ctx) -> None:
        """Refresh credentials and re-inject auth headers."""
        await self.authenticator.refresh_credentials(creds_id=creds_id)
        await self._inject_auth_headers(ctx)


_RETRYABLE_AUTH_CODES = frozenset({Code.UNAUTHENTICATED, Code.UNKNOWN})


class AuthUnaryInterceptor(_BaseAuthInterceptor):
    """ConnectRPC unary interceptor that injects auth headers and retries on UNAUTHENTICATED."""

    async def intercept_unary(self, call_next, request, ctx):
        creds_id = await self._inject_auth_headers(ctx)
        try:
            return await call_next(request, ctx)
        except ConnectError as e:
            if e.code in _RETRYABLE_AUTH_CODES:
                await self._refresh_and_reinject(creds_id, ctx)
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
        creds_id = await self._inject_auth_headers(ctx)
        try:
            return await call_next(request, ctx)
        except ConnectError as e:
            if e.code in _RETRYABLE_AUTH_CODES:
                await self._refresh_and_reinject(creds_id, ctx)
                return await call_next(request, ctx)
            raise


class AuthServerStreamInterceptor(_BaseAuthInterceptor):
    """ConnectRPC server-stream interceptor that injects auth headers and retries on UNAUTHENTICATED."""

    async def intercept_server_stream(self, call_next, request, ctx):
        creds_id = await self._inject_auth_headers(ctx)
        try:
            async for response in call_next(request, ctx):
                yield response
        except ConnectError as e:
            if e.code in _RETRYABLE_AUTH_CODES:
                await self._refresh_and_reinject(creds_id, ctx)
                async for response in call_next(request, ctx):
                    yield response
            else:
                raise


class AuthBidiStreamInterceptor(_BaseAuthInterceptor):
    """ConnectRPC bidi-stream interceptor that injects auth headers and retries on UNAUTHENTICATED.

    See AuthClientStreamInterceptor for the request-iterator replay caveat.
    """

    async def intercept_bidi_stream(self, call_next, request, ctx):
        creds_id = await self._inject_auth_headers(ctx)
        try:
            async for response in call_next(request, ctx):
                yield response
        except ConnectError as e:
            if e.code in _RETRYABLE_AUTH_CODES:
                await self._refresh_and_reinject(creds_id, ctx)
                async for response in call_next(request, ctx):
                    yield response
            else:
                raise
