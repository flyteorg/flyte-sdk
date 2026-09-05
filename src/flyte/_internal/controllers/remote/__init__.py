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
    scopes: List[str] | None = None,
    rpc_retries: int = 3,
    http_proxy_url: str | None = None,
) -> RemoteController:
    """
    Create a new instance of the remote controller.
    """
    assert endpoint or api_key, "Either endpoint or api_key must be provided when initializing remote controller"
    from ._client import ControllerClient
    from ._controller import RemoteController

    # Keep this set in sync with `_initialize._initialize_client`: the controller and the
    # SDK client are built from the same kwargs (see `init_in_cluster`), so an auth field
    # dropped here authenticates the client but leaves the controller unable to enqueue.
    auth_kwargs: dict = {
        "insecure": insecure,
        "insecure_skip_verify": insecure_skip_verify,
        "ca_cert_file_path": ca_cert_file_path,
        "client_id": client_id,
        "client_credentials_secret": client_credentials_secret,
        "scopes": scopes,
        "auth_type": auth_type,
        "command": command,
        "proxy_command": proxy_command,
        "http_proxy_url": http_proxy_url,
        "client_config": client_config,
    }

    if endpoint:
        client_coro = ControllerClient.for_endpoint(endpoint, **auth_kwargs)
    elif api_key:
        client_coro = ControllerClient.for_api_key(api_key, **auth_kwargs)

    controller = RemoteController(
        client_coro=client_coro,
    )
    return controller
