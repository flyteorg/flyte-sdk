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

    if endpoint:
        client_coro = ControllerClient.for_endpoint(
            endpoint,
            insecure=insecure,
            insecure_skip_verify=insecure_skip_verify,
            ca_cert_file_path=ca_cert_file_path,
            client_id=client_id,
            client_credentials_secret=client_credentials_secret,
            auth_type=auth_type,
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
        )

    controller = RemoteController(
        client_coro=client_coro,
    )
    return controller
