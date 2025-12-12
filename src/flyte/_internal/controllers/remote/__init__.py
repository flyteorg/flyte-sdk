from typing import List

from flyte.remote._client.auth import AuthType, ClientConfig

from ._controller import RemoteController

__all__ = ["RemoteController", "create_remote_controller"]


def create_remote_controller(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    insecure: bool = False,
    insecure_skip_verify: bool = False,
    ca_cert_file_path: str | None = None,
    client_config: ClientConfig | None = None,
    auth_type: AuthType = "Pkce",
    headless: bool = False,
    command: List[str] | None = None,
    proxy_command: List[str] | None = None,
    client_id: str | None = None,
    client_credentials_secret: str | None = None,
    rpc_retries: int = 3,
    http_proxy_url: str | None = None,
) -> RemoteController:
    """
    Create a new instance of the remote controller.
    """
    assert endpoint or api_key, "Either endpoint or api_key must be provided when initializing remote controller"
    from ._client import ControllerClient
    from ._controller import RemoteController

    # https://grpc.io/docs/guides/keepalive/#keepalive-configuration-specification
    channel_options = [
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.keepalive_time_ms", 30000),  # Send keepalive ping every 30 seconds
        ("grpc.keepalive_timeout_ms", 10000),  # Wait 10 seconds for keepalive response
        ("grpc.http2.max_pings_without_data", 0),  # Allow unlimited pings without data
        ("grpc.http2.min_ping_interval_without_data_ms", 30000),  # Min 30s between pings
    ]

    if endpoint:
        client_coro = ControllerClient.for_endpoint(
            endpoint,
            insecure=insecure,
            insecure_skip_verify=insecure_skip_verify,
            ca_cert_file_path=ca_cert_file_path,
            client_id=client_id,
            client_credentials_secret=client_credentials_secret,
            auth_type=auth_type,
            grpc_options=channel_options,
        )
    elif api_key:
        client_coro = ControllerClient.for_api_key(
            api_key,
            insecure=insecure,
            insecure_skip_verify=insecure_skip_verify,
            ca_cert_file_path=ca_cert_file_path,
            client_id=client_id,
            client_credentials_secret=client_credentials_secret,
            auth_type=auth_type,
            grpc_options=channel_options,
        )

    controller = RemoteController(
        client_coro=client_coro,
    )
    return controller
