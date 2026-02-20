import asyncio
import random
import typing
from typing import AsyncIterator, Optional, Set, Union

import grpc
import grpc.aio
from grpc.aio import ClientCallDetails, Metadata
from grpc.aio._typing import DoneCallbackType, EOFType, RequestType, ResponseType

from flyte._logging import logger

_RETRYABLE_STATUS_CODES: Set[grpc.StatusCode] = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.INTERNAL,
}


def _is_retryable(code: grpc.StatusCode) -> bool:
    return code in _RETRYABLE_STATUS_CODES


class _BaseRetryInterceptor:
    """Base class for retry interceptors with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        min_backoff: float = 0.5,
        max_backoff: float = 10.0,
    ):
        self._max_retries = max_retries
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff

    def _backoff(self, attempt: int) -> float:
        base = min(self._min_backoff * (2**attempt), self._max_backoff)
        jitter = base * random.uniform(0, 0.25)
        return base + jitter


class RetryUnaryUnaryInterceptor(_BaseRetryInterceptor, grpc.aio.UnaryUnaryClientInterceptor):
    """Interceptor that retries unary-unary calls on transient gRPC errors."""

    async def intercept_unary_unary(
        self,
        continuation: typing.Callable,
        client_call_details: ClientCallDetails,
        request: typing.Any,
    ):
        for attempt in range(1 + self._max_retries):
            try:
                return await (await continuation(client_call_details, request))
            except grpc.aio.AioRpcError as e:
                if not _is_retryable(e.code()) or attempt >= self._max_retries:
                    raise
                delay = self._backoff(attempt)
                logger.warning(
                    f"gRPC call {client_call_details.method} failed with {e.code().name}, "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self._max_retries})"
                )
                await asyncio.sleep(delay)

        raise RuntimeError("unreachable")  # linting


class _RetryUnaryStreamCall(grpc.aio.UnaryStreamCall):
    """Wrapper around a unary-stream call that retries on transient errors."""

    def __init__(
        self,
        interceptor: _BaseRetryInterceptor,
        continuation: typing.Callable,
        call_details: ClientCallDetails,
        request: RequestType,
    ):
        super().__init__()
        self._interceptor = interceptor
        self._continuation = continuation
        self._call_details = call_details
        self._request = request
        self._call: (
            Union[
                grpc.aio.UnaryStreamCall[RequestType, ResponseType],
                grpc.aio.StreamStreamCall[RequestType, ResponseType],
            ]
            | None
        ) = None

    async def _response_iterator(self) -> typing.AsyncIterator[ResponseType]:
        attempt = 0
        while True:
            self._call = await self._continuation(self._call_details, self._request)
            try:
                async for response in self._call:
                    attempt = 0  # reset on successful message
                    yield response
                return  # stream completed normally
            except grpc.aio.AioRpcError as e:
                if not _is_retryable(e.code()) or attempt >= self._interceptor._max_retries:
                    raise
                delay = self._interceptor._backoff(attempt)
                logger.warning(
                    f"gRPC stream {self._call_details.method} failed with {e.code().name}, "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self._interceptor._max_retries})"
                )
                await asyncio.sleep(delay)
                attempt += 1

    def __aiter__(self) -> AsyncIterator[ResponseType]:
        return self._response_iterator()

    async def read(self) -> Union[EOFType, ResponseType]:
        if self._call is not None:
            return await self._call.read()
        return EOFType()

    async def initial_metadata(self) -> Metadata:
        if self._call is not None:
            return await self._call.initial_metadata()
        return Metadata()

    async def trailing_metadata(self) -> Metadata:
        if self._call is not None:
            return await self._call.trailing_metadata()
        return Metadata()

    async def code(self) -> grpc.StatusCode:
        if self._call is not None:
            return await self._call.code()
        return grpc.StatusCode.OK

    async def details(self) -> str:
        if self._call is not None:
            return await self._call.details()
        return ""

    async def wait_for_connection(self) -> None:
        if self._call is not None:
            await self._call.wait_for_connection()
        return None

    def cancelled(self) -> bool:
        if self._call is not None:
            return self._call.cancelled()
        return False

    def done(self) -> bool:
        if self._call is not None:
            return self._call.done()
        return False

    def time_remaining(self) -> Optional[float]:
        if self._call is not None:
            return self._call.time_remaining()
        return None

    def cancel(self) -> bool:
        if self._call is not None:
            return self._call.cancel()
        return False

    def add_done_callback(self, callback: DoneCallbackType) -> None:
        if self._call is not None:
            self._call.add_done_callback(callback=callback)
        return None


class RetryUnaryStreamInterceptor(_BaseRetryInterceptor, grpc.aio.UnaryStreamClientInterceptor):
    """Interceptor that retries unary-stream calls on transient gRPC errors."""

    async def intercept_unary_stream(
        self,
        continuation: typing.Callable,
        client_call_details: ClientCallDetails,
        request: typing.Any,
    ):
        return _RetryUnaryStreamCall(
            interceptor=self,
            continuation=continuation,
            call_details=client_call_details,
            request=request,
        )
