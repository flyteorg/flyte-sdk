import random
import string
import typing
from uuid import uuid4

import grpc.aio
from grpc.aio import ClientCallDetails, Metadata

import flyte

_default_metadata = Metadata(("accept", "application/grpc"))
ALPHABET = string.ascii_lowercase + string.digits


def fast_short_id():
    return "".join(random.choices(ALPHABET, k=4))


def _generate_request_id() -> str:
    """
    Generate a request ID based on the current Flyte context.

    If running within a Flyte task context, creates a request ID using the action's unique_id_str method.
    Otherwise, falls back to a UUID4.

    :return: The generated request ID string
    """
    ctx = flyte.ctx()
    if ctx and ctx.action:
        return ctx.action.unique_id_str(salt=fast_short_id())

    # Fall back to UUID4 if context is not available
    return str(uuid4())


def with_metadata(call_details: ClientCallDetails, new_metadata: Metadata) -> ClientCallDetails:
    metadata = Metadata()
    for k, v in call_details.metadata.keys():
        # Add existing metadata to the new metadata object
        metadata.add(key=k, value=v)
    for k, v in new_metadata.keys():
        metadata.add(key=k, value=v)

    # return call_details._replace(metadata=metadata), None
    return ClientCallDetails(
        method=call_details.method,
        timeout=call_details.timeout,
        metadata=metadata,
        credentials=call_details.credentials,
        wait_for_ready=call_details.wait_for_ready,
    )


class _BaseDefaultMetadataInterceptor:
    """
    Base class for all default metadata interceptors that provides common functionality.
    Injects both default metadata (accept header) and x-request-id.
    """

    async def _inject_default_metadata(self, call_details: grpc.aio.ClientCallDetails):
        """
        Injects default metadata and request ID into the client call details.

        This method adds:
        - Default metadata (accept: application/grpc)
        - x-request-id header with context-based or UUID4 value

        :param call_details: The client call details to inject metadata into
        :return: A new ClientCallDetails object with the injected metadata
        """
        # Generate request ID and combine with default metadata
        request_id = _generate_request_id()
        combined_metadata = Metadata(
            ("accept", "application/grpc"),
            ("x-request-id", request_id),
        )
        return with_metadata(call_details, combined_metadata)


class DefaultMetadataUnaryUnaryInterceptor(_BaseDefaultMetadataInterceptor, grpc.aio.UnaryUnaryClientInterceptor):
    """
    Interceptor for unary-unary RPC calls that adds default metadata.
    """

    async def intercept_unary_unary(
        self,
        continuation: typing.Callable,
        client_call_details: grpc.aio.ClientCallDetails,
        request: typing.Any,
    ):
        """
        Intercepts unary-unary calls and injects default metadata.

        This method adds default metadata to the client call details before continuing the RPC call chain.

        :param continuation: Function to continue the RPC call chain with the updated call details
        :param client_call_details: Details about the RPC call including method, timeout, metadata, credentials,
        and wait_for_ready
        :param request: The request message to be sent to the server
        :return: The response from the RPC call
        """
        updated_call_details = await self._inject_default_metadata(client_call_details)
        return await (await continuation(updated_call_details, request))


class DefaultMetadataUnaryStreamInterceptor(_BaseDefaultMetadataInterceptor, grpc.aio.UnaryStreamClientInterceptor):
    """
    Interceptor for unary-stream RPC calls that adds default metadata.
    """

    async def intercept_unary_stream(
        self,
        continuation: typing.Callable,
        client_call_details: grpc.aio.ClientCallDetails,
        request: typing.Any,
    ):
        """
        Intercepts unary-stream calls and injects default metadata.

        This method adds default metadata to the client call details before continuing the RPC call chain.

        :param continuation: Function to continue the RPC call chain with the updated call details
        :param client_call_details: Details about the RPC call including method, timeout, metadata, credentials,
        and wait_for_ready
        :param request: The request message to be sent to the server
        :return: A stream of responses from the RPC call
        """
        updated_call_details = await self._inject_default_metadata(client_call_details)
        return await continuation(updated_call_details, request)


class DefaultMetadataStreamUnaryInterceptor(_BaseDefaultMetadataInterceptor, grpc.aio.StreamUnaryClientInterceptor):
    """
    Interceptor for stream-unary RPC calls that adds default metadata.
    """

    async def intercept_stream_unary(
        self,
        continuation: typing.Callable,
        client_call_details: grpc.aio.ClientCallDetails,
        request_iterator: typing.Any,
    ):
        """
        Intercepts stream-unary calls and injects default metadata.

        This method adds default metadata to the client call details before continuing the RPC call chain.

        :param continuation: Function to continue the RPC call chain with the updated call details
        :param client_call_details: Details about the RPC call including method, timeout, metadata, credentials,
        and wait_for_ready
        :param request_iterator: An iterator of request messages to be sent to the server
        :return: The response from the RPC call
        """
        updated_call_details = await self._inject_default_metadata(client_call_details)
        return await continuation(updated_call_details, request_iterator)


class DefaultMetadataStreamStreamInterceptor(_BaseDefaultMetadataInterceptor, grpc.aio.StreamStreamClientInterceptor):
    """
    Interceptor for stream-stream RPC calls that adds default metadata.
    """

    async def intercept_stream_stream(
        self,
        continuation: typing.Callable,
        client_call_details: grpc.aio.ClientCallDetails,
        request_iterator: typing.Any,
    ):
        """
        Intercepts stream-stream calls and injects default metadata.

        This method adds default metadata to the client call details before continuing the RPC call chain.

        :param continuation: Function to continue the RPC call chain with the updated call details
        :param client_call_details: Details about the RPC call including method, timeout, metadata, credentials, and
         wait_for_ready
        :param request_iterator: An iterator of request messages to be sent to the server
        :return: A stream of responses from the RPC call
        """
        updated_call_details = await self._inject_default_metadata(client_call_details)
        return await continuation(updated_call_details, request_iterator)


# For backward compatibility, maintain the original class name but as a type alias
DefaultMetadataInterceptor = DefaultMetadataUnaryUnaryInterceptor
