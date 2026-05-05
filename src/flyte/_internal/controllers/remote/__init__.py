import os
from typing import List

from flyte._logging import logger
from flyte.remote._client.auth import AuthType, ClientConfig

from ._controller import RemoteController

__all__ = ["RemoteController", "create_remote_controller"]

# Env var that opts into the Rust-backed RemoteController (flyte_controller_base
# wheel). Off by default — the Rust controller does not yet support the unified
# Actions service path; if _U_USE_ACTIONS=1 is set, we always fall back to the
# Python implementation regardless of this flag.
_USE_RUST_CONTROLLER_ENV_VAR = "_F_USE_RUST_CONTROLLER"


def _use_rust_controller() -> bool:
    if os.getenv("_U_USE_ACTIONS") == "1":
        # Actions service path is Python-only until flyteidl2 Rust crate exposes it.
        return False
    return os.getenv(_USE_RUST_CONTROLLER_ENV_VAR) == "1"


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

    Selection between the pure-Python implementation and the Rust-backed
    implementation (``flyte_controller_base``) is controlled by the
    ``_F_USE_RUST_CONTROLLER`` environment variable. The Rust path is only
    selected when ``flyte_controller_base`` is importable; otherwise we
    log a warning and fall back to the Python implementation.
    """
    assert endpoint or api_key, "Either endpoint or api_key must be provided when initializing remote controller"

    if _use_rust_controller():
        try:
            import flyte_controller_base  # noqa: F401  (probe import)

            from ._r_controller import RemoteController as RustRemoteController

            logger.info("Using Rust-backed RemoteController (flyte_controller_base).")
            # The Rust controller manages its own gRPC channel, auth, and
            # informer loop. It only needs the endpoint; api-key auth is
            # picked up from the FLYTE_API_KEY env var by the Rust side.
            return RustRemoteController(endpoint=endpoint)
        except ImportError as e:
            logger.warning(
                f"_F_USE_RUST_CONTROLLER=1 was set but flyte_controller_base is not "
                f"importable ({e}); falling back to the Python RemoteController."
            )

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
