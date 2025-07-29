import typing
from typing import AsyncIterator, Optional, Union

import grpc.aio
from grpc.aio import ClientCallDetails, Metadata
from grpc.aio._typing import DoneCallbackType, EOFType, RequestType, ResponseType

from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._grpc_utils.default_metadata_interceptor import with_metadata


class _BaseAuthInterceptor:
    """
    Base class for all auth interceptors that provides common authentication functionality.
    """

    def __init__(self, get_authenticator: typing.Callable[[], Authenticator]):
        self._get_authenticator = get_authenticator
        self._authenticator: typing.Optional[Authenticator] = None

    @property
    def authenticator(self) -> Authenticator:
        if self._authenticator is None:
            self._authenticator = self._get_authenticator()
        return self._authenticator

    async def call_details_with_auth_metadata(
        self, client_call_details: grpc.aio.ClientCallDetails
    ) -> typing.Tuple[grpc.aio.ClientCallDetails, str]:
        """
        Returns new ClientCallDetails with authentication metadata added.

        This method retrieves authentication metadata from the authenticator and adds it to the
        client call details. If no authentication metadata is available, the original client call
        details are returned unchanged.

        :param client_call_details: The original client call details containing method, timeout, metadata,
                                   credentials, and wait_for_ready settings
        :return: Updated client call details with authentication metadata added to the existing metadata
        """
        auth_metadata = await self.authenticator.get_grpc_call_auth_metadata()
        if auth_metadata:
            return with_metadata(client_call_details, auth_metadata.pairs), auth_metadata.creds_id
        else:
            return client_call_details, ""


class AuthUnaryUnaryInterceptor(_BaseAuthInterceptor, grpc.aio.UnaryUnaryClientInterceptor):
    """
    Interceptor for unary-unary RPC calls that adds authentication metadata.
    """

    async def intercept_unary_unary(
        self,
        continuation: typing.Callable,
        client_call_details: ClientCallDetails,
        request: typing.Any,
    ):
        """
        Intercepts unary-unary calls and adds auth metadata if available. On Unauthenticated, resets the token and
        refreshes and then retries with the new token.

        This method first adds authentication metadata to the client call details, then attempts to make the RPC call.
        If the call fails with an UNAUTHENTICATED or UNKNOWN status code, it refreshes the credentials and retries
        the call with the new authentication metadata.

        :param continuation: Function to continue the RPC call chain with the updated call details
        :param client_call_details: Details about the RPC call including method, timeout, metadata, credentials,
        and wait_for_ready
        :param request: The request message to be sent to the server
        :return: The response from the RPC call after successful authentication
        :raises: grpc.aio.AioRpcError if the call fails for reasons other than authentication
        """
        updated_call_details, creds_id = await self.call_details_with_auth_metadata(client_call_details)
        try:
            return await (await continuation(updated_call_details, request))
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED or e.code() == grpc.StatusCode.UNKNOWN:
                await self.authenticator.refresh_credentials(creds_id=creds_id)
                updated_call_details, _ = await self.call_details_with_auth_metadata(client_call_details)
                return await (await continuation(updated_call_details, request))
            else:
                raise e


class UnaryStreamCall(grpc.aio.UnaryStreamCall):
    def __init__(
        self,
        parent_interceptor: _BaseAuthInterceptor,
        authenticator: Authenticator,
        continuation: typing.Callable,
        call_details: grpc.aio.ClientCallDetails,
        request: RequestType,
    ):
        super().__init__()
        self._continuation = continuation
        self._call_details = call_details
        self._request = request
        self._authenticator = authenticator
        self._parent_interceptor = parent_interceptor
        self._call: (
            Union[
                grpc.aio.UnaryStreamCall[RequestType, ResponseType],
                grpc.aio.StreamStreamCall[RequestType, ResponseType],
            ]
            | None
        ) = None

    async def response_iterator(self) -> typing.AsyncIterator[ResponseType]:
        call_details, creds_id = await self._parent_interceptor.call_details_with_auth_metadata(self._call_details)
        self._call = await self._continuation(call_details, self._request)
        try:
            async for response in self._call:
                yield response
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED or e.code() == grpc.StatusCode.UNKNOWN:
                await self._authenticator.refresh_credentials(creds_id=creds_id)
                updated_call_details, _ = await self._parent_interceptor.call_details_with_auth_metadata(call_details)
                self._call = await self._continuation(updated_call_details, self._request)
                async for response in self._call:
                    yield response
            else:
                raise e

    def __aiter__(self) -> AsyncIterator[ResponseType]:
        return self.response_iterator()

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


class AuthUnaryStreamInterceptor(_BaseAuthInterceptor, grpc.aio.UnaryStreamClientInterceptor):
    """
    Interceptor for unary-stream RPC calls that adds authentication metadata.
    """

    async def intercept_unary_stream(
        self, continuation: typing.Callable, client_call_details: grpc.aio.ClientCallDetails, request: typing.Any
    ):
        """
        Intercepts unary-stream calls and adds auth metadata if available.

        This method first adds authentication metadata to the client call details, then attempts to make the RPC call.
        If the call fails with an UNAUTHENTICATED or UNKNOWN status code, it refreshes the credentials and retries
        the call with the new authentication metadata.

        :param continuation: Function to continue the RPC call chain with the updated call details
        :param client_call_details: Details about the RPC call including method, timeout, metadata, credentials,
        and wait_for_ready
        :param request: The request message to be sent to the server
        :return: A stream of responses from the RPC call after successful authentication
        :raises: grpc.aio.AioRpcError if the call fails for reasons other than authentication
        """

        return UnaryStreamCall(
            parent_interceptor=self,
            authenticator=self.authenticator,
            call_details=client_call_details,
            continuation=continuation,
            request=request,
        )


class AuthStreamUnaryInterceptor(_BaseAuthInterceptor, grpc.aio.StreamUnaryClientInterceptor):
    """
    Interceptor for stream-unary RPC calls that adds authentication metadata.
    """

    async def intercept_stream_unary(
        self,
        continuation: typing.Callable,
        client_call_details: grpc.aio.ClientCallDetails,
        request_iterator: typing.Any,
    ):
        """
        Intercepts stream-unary calls and adds auth metadata if available.

        This method first adds authentication metadata to the client call details, then attempts to make the RPC call.
        If the call fails with an UNAUTHENTICATED or UNKNOWN status code, it refreshes the credentials and retries
        the call with the new authentication metadata.

        :param continuation: Function to continue the RPC call chain with the updated call details
        :param client_call_details: Details about the RPC call including method, timeout, metadata, credentials,
        and wait_for_ready
        :param request_iterator: An iterator of request messages to be sent to the server
        :return: The response from the RPC call after successful authentication
        :raises: grpc.aio.AioRpcError if the call fails for reasons other than authentication
        """
        updated_call_details, creds_id = await self.call_details_with_auth_metadata(client_call_details)
        try:
            call = await continuation(updated_call_details, request_iterator)
            return await call
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED or e.code() == grpc.StatusCode.UNKNOWN:
                await self.authenticator.refresh_credentials(creds_id=creds_id)
                updated_call_details, _ = await self.call_details_with_auth_metadata(client_call_details)
                call = await continuation(updated_call_details, request_iterator)
                return await call
            raise e


class AuthStreamStreamInterceptor(_BaseAuthInterceptor, grpc.aio.StreamStreamClientInterceptor):
    """
    Interceptor for stream-stream RPC calls that adds authentication metadata.
    """

    async def intercept_stream_stream(
        self,
        continuation: typing.Callable,
        client_call_details: grpc.aio.ClientCallDetails,
        request_iterator: typing.Any,
    ):
        """
        Intercepts stream-stream calls and adds auth metadata if available.

        This method first adds authentication metadata to the client call details, then attempts to make the RPC call.
        If the call fails with an UNAUTHENTICATED or UNKNOWN status code, it refreshes the credentials and retries
        the call with the new authentication metadata.

        :param continuation: Function to continue the RPC call chain with the updated call details
        :param client_call_details: Details about the RPC call including method, timeout, metadata, credentials,
        and wait_for_ready
        :param request_iterator: An iterator of request messages to be sent to the server
        :return: A stream of responses from the RPC call after successful authentication
        """
        return UnaryStreamCall(
            parent_interceptor=self,
            authenticator=self.authenticator,
            call_details=client_call_details,
            continuation=continuation,
            request=request_iterator,
        )


# For backward compatibility, maintain the original class name but as a type alias
AuthUnaryInterceptor = AuthUnaryUnaryInterceptor
