import typing

from grpc.aio import Metadata

from flyte.remote._client.auth._authenticators.base import Authenticator, GrpcAuthMetadata
from flyte.remote._client.auth._keyring import Credentials


class PassthroughAuthenticator(Authenticator):
    """
    Passthrough authenticator that extracts headers from the context and passes them
    to the gRPC calls without performing any authentication flow.

    This authenticator is used when you want to pass custom authentication metadata
    using the flyte.remote.auth_metadata() context manager.
    """

    def __init__(self, endpoint: str, **kwargs):
        """
        Initialize the passthrough authenticator.

        :param endpoint: The endpoint URL
        :param kwargs: Additional arguments (ignored for passthrough auth)
        """
        # Don't call parent __init__ to avoid unnecessary setup for passthrough auth
        self._endpoint = endpoint
        # We don't need credentials, config store, or HTTP session for passthrough
        self._creds = None
        self._creds_id: str = "passthrough"

    def get_credentials(self) -> typing.Optional[Credentials]:
        """
        Passthrough authenticator doesn't use traditional credentials.
        Returns a dummy credential to signal that metadata is available.
        """
        # Return a dummy credential so the interceptor knows to call get_grpc_call_auth_metadata
        return Credentials(
            access_token="passthrough",
            for_endpoint=self._endpoint,
        )

    async def get_grpc_call_auth_metadata(self) -> typing.Optional[GrpcAuthMetadata]:
        """
        Fetch the authentication metadata from the context.

        :return: GrpcAuthMetadata with the metadata from the context, or None if no metadata is available
        """
        # Lazy import to avoid circular dependencies
        from flyte.remote._auth_metadata import get_auth_metadata

        # Get metadata from context
        metadata_tuples = get_auth_metadata()

        if not metadata_tuples:
            return None

        # Print the metadata for debugging (as requested)
        print(f"[PassthroughAuthenticator] Passing metadata: {metadata_tuples}")
        for key, value in metadata_tuples:
            print(f"  {key}: {value}")

        # Convert tuples to gRPC Metadata
        metadata = Metadata(*metadata_tuples)

        return GrpcAuthMetadata(
            creds_id=self._creds_id,
            pairs=metadata,
        )

    async def _do_refresh_credentials(self) -> Credentials:
        """
        Passthrough authenticator doesn't need to refresh credentials.
        This method should never be called in practice.
        """
        # Return a dummy credential
        return Credentials(
            access_token="passthrough",
            for_endpoint=self._endpoint,
        )
