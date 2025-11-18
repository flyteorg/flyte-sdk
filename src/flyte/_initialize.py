from __future__ import annotations

import functools
import threading
import typing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, TypeVar

from flyte.errors import InitializationError
from flyte.syncify import syncify

from ._logging import LogFormat, initialize_logger, logger

if TYPE_CHECKING:
    from flyte._internal.imagebuild import ImageBuildEngine
    from flyte.config import Config
    from flyte.remote._client.auth import AuthType, ClientConfig
    from flyte.remote._client.controlplane import ClientSet
    from flyte.storage import Storage

Mode = Literal["local", "remote"]


@dataclass(init=True, repr=True, eq=True, frozen=True, kw_only=True)
class CommonInit:
    """
    Common initialization configuration for Flyte.
    """

    root_dir: Path
    org: str | None = None
    project: str | None = None
    domain: str | None = None
    batch_size: int = 1000
    source_config_path: Optional[Path] = None  # Only used for documentation
    sync_local_sys_paths: bool = True


@dataclass(init=True, kw_only=True, repr=True, eq=True, frozen=True)
class _InitConfig(CommonInit):
    client: Optional[ClientSet] = None
    storage: Optional[Storage] = None
    image_builder: "ImageBuildEngine.ImageBuilderType" = "local"
    images: typing.Dict[str, str] = field(default_factory=dict)

    def replace(self, **kwargs) -> _InitConfig:
        return replace(self, **kwargs)


# Global singleton to store initialization configuration
_init_config: _InitConfig | None = None
_init_lock = threading.RLock()  # Reentrant lock for thread safety


async def _initialize_client(
    api_key: str | None = None,
    auth_type: AuthType = "Pkce",
    endpoint: str | None = None,
    client_config: ClientConfig | None = None,
    headless: bool = False,
    insecure: bool = False,
    insecure_skip_verify: bool = False,
    ca_cert_file_path: str | None = None,
    command: List[str] | None = None,
    proxy_command: List[str] | None = None,
    client_id: str | None = None,
    client_credentials_secret: str | None = None,
    rpc_retries: int = 3,
    http_proxy_url: str | None = None,
) -> ClientSet:
    """
    Initialize the client based on the execution mode.
    :return: The initialized client
    """
    from flyte.remote._client.controlplane import ClientSet

    if endpoint:
        return await ClientSet.for_endpoint(
            endpoint,
            insecure=insecure,
            insecure_skip_verify=insecure_skip_verify,
            auth_type=auth_type,
            headless=headless,
            ca_cert_file_path=ca_cert_file_path,
            command=command,
            proxy_command=proxy_command,
            client_id=client_id,
            client_credentials_secret=client_credentials_secret,
            client_config=client_config,
            rpc_retries=rpc_retries,
            http_proxy_url=http_proxy_url,
        )
    elif api_key:
        return await ClientSet.for_api_key(
            api_key,
            insecure=insecure,
            insecure_skip_verify=insecure_skip_verify,
            auth_type=auth_type,
            headless=headless,
            ca_cert_file_path=ca_cert_file_path,
            command=command,
            proxy_command=proxy_command,
            client_id=client_id,
            client_credentials_secret=client_credentials_secret,
            client_config=client_config,
            rpc_retries=rpc_retries,
            http_proxy_url=http_proxy_url,
        )

    raise InitializationError(
        "MissingEndpointOrApiKeyError", "user", "Either endpoint or api_key must be provided to initialize the client."
    )


def _initialize_logger(log_level: int | None = None, log_format: LogFormat | None = None) -> None:
    initialize_logger(log_level=log_level, log_format=log_format, enable_rich=True)


@syncify
async def init(
    org: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    root_dir: Path | None = None,
    log_level: int | None = None,
    log_format: LogFormat | None = None,
    endpoint: str | None = None,
    headless: bool = False,
    insecure: bool = False,
    insecure_skip_verify: bool = False,
    ca_cert_file_path: str | None = None,
    auth_type: AuthType = "Pkce",
    command: List[str] | None = None,
    proxy_command: List[str] | None = None,
    api_key: str | None = None,
    client_id: str | None = None,
    client_credentials_secret: str | None = None,
    auth_client_config: ClientConfig | None = None,
    rpc_retries: int = 3,
    http_proxy_url: str | None = None,
    storage: Storage | None = None,
    batch_size: int = 1000,
    image_builder: ImageBuildEngine.ImageBuilderType = "local",
    images: typing.Dict[str, str] | None = None,
    source_config_path: Optional[Path] = None,
    sync_local_sys_paths: bool = True,
    load_plugin_type_transformers: bool = True,
) -> None:
    """
    Initialize the Flyte system with the given configuration. This method should be called before any other Flyte
    remote API methods are called. Thread-safe implementation.

    :param project: Optional project name (not used in this implementation)
    :param domain: Optional domain name (not used in this implementation)
    :param root_dir: Optional root directory from which to determine how to load files, and find paths to files.
      This is useful for determining the root directory for the current project, and for locating files like config etc.
      also use to determine all the code that needs to be copied to the remote location.
      defaults to the editable install directory if the cwd is in a Python editable install, else just the cwd.
    :param log_level: Optional logging level for the logger, default is set using the default initialization policies
    :param log_format: Optional logging format for the logger, default is "console"
    :param api_key: Optional API key for authentication
    :param endpoint: Optional API endpoint URL
    :param headless: Optional Whether to run in headless mode
    :param insecure_skip_verify: Whether to skip SSL certificate verification
    :param auth_client_config: Optional client configuration for authentication
    :param auth_type: The authentication type to use (Pkce, ClientSecret, ExternalCommand, DeviceFlow)
    :param command: This command is executed to return a token using an external process
    :param proxy_command: This command is executed to return a token for proxy authorization using an external process
    :param client_id: This is the public identifier for the app which handles authorization for a Flyte deployment.
      More details here: https://www.oauth.com/oauth2-servers/client-registration/client-id-secret/.
    :param client_credentials_secret: Used for service auth, which is automatically called during pyflyte. This will
      allow the Flyte engine to read the password directly from the environment variable. Note that this is
      less secure! Please only use this if mounting the secret as a file is impossible
    :param ca_cert_file_path: [optional] str Root Cert to be loaded and used to verify admin
    :param http_proxy_url: [optional] HTTP Proxy to be used for OAuth requests
    :param rpc_retries: [optional] int Number of times to retry the platform calls
    :param insecure: insecure flag for the client
    :param storage: Optional blob store (S3, GCS, Azure) configuration if needed to access (i.e. using Minio)
    :param org: Optional organization override for the client. Should be set by auth instead.
    :param batch_size: Optional batch size for operations that use listings, defaults to 1000, so limit larger than
      batch_size will be split into multiple requests.
    :param image_builder: Optional image builder configuration, if not provided, the default image builder will be used.
    :param images: Optional dict of images that can be used by referencing the image name.
    :param source_config_path: Optional path to the source configuration file (This is only used for documentation)
    :param sync_local_sys_paths: Whether to include and synchronize local sys.path entries under the root directory
      into the remote container (default: True).
    :param load_plugin_type_transformers: If enabled (default True), load the type transformer plugins registered under
      the "flyte.plugins.types" entry point group.
    :return: None
    """
    from flyte._utils import get_cwd_editable_install, org_from_endpoint, sanitize_endpoint
    from flyte.types import _load_custom_type_transformers

    _initialize_logger(log_level=log_level, log_format=log_format)
    if load_plugin_type_transformers:
        _load_custom_type_transformers()

    global _init_config  # noqa: PLW0603

    endpoint = sanitize_endpoint(endpoint)

    with _init_lock:
        client = None
        if endpoint or api_key:
            client = await _initialize_client(
                api_key=api_key,
                auth_type=auth_type,
                endpoint=endpoint,
                headless=headless,
                insecure=insecure,
                insecure_skip_verify=insecure_skip_verify,
                ca_cert_file_path=ca_cert_file_path,
                command=command,
                proxy_command=proxy_command,
                client_id=client_id,
                client_credentials_secret=client_credentials_secret,
                client_config=auth_client_config,
                rpc_retries=rpc_retries,
                http_proxy_url=http_proxy_url,
            )

        if not root_dir:
            editable_root = get_cwd_editable_install()
            if editable_root:
                logger.info(f"Using editable install as root directory: {editable_root}")
                root_dir = editable_root
            else:
                logger.info("No editable install found, using current working directory as root directory.")
                root_dir = Path.cwd()

        _init_config = _InitConfig(
            root_dir=root_dir,
            project=project,
            domain=domain,
            client=client,
            storage=storage,
            org=org or org_from_endpoint(endpoint),
            batch_size=batch_size,
            image_builder=image_builder,
            images=images or {},
            source_config_path=source_config_path,
            sync_local_sys_paths=sync_local_sys_paths,
        )


@syncify
async def init_from_config(
    path_or_config: str | Path | Config | None = None,
    root_dir: Path | None = None,
    log_level: int | None = None,
    log_format: LogFormat = "console",
    storage: Storage | None = None,
    images: tuple[str, ...] | None = None,
    sync_local_sys_paths: bool = True,
) -> None:
    """
    Initialize the Flyte system using a configuration file or Config object. This method should be called before any
    other Flyte remote API methods are called. Thread-safe implementation.

    :param path_or_config: Path to the configuration file or Config object
    :param root_dir: Optional root directory from which to determine how to load files, and find paths to
        files like config etc. For example if one uses the copy-style=="all", it is essential to determine the
        root directory for the current project. If not provided, it defaults to the editable install directory or
        if not available, the current working directory.
    :param log_level: Optional logging level for the framework logger,
        default is set using the default initialization policies
    :param log_format: Optional logging format for the logger, default is "console"
    :param storage: Optional blob store (S3, GCS, Azure) configuration if needed to access (i.e. using Minio)
    :param images: List of image strings in format "imagename=imageuri" or just "imageuri".
    :param sync_local_sys_paths: Whether to include and synchronize local sys.path entries under the root directory
     into the remote container (default: True).
    :return: None
    """
    from rich.highlighter import ReprHighlighter

    import flyte.config as config
    from flyte.cli._common import parse_images

    cfg: config.Config
    cfg_path: Optional[Path] = None
    if path_or_config is None:
        # If no path is provided, use the default config file
        cfg = config.auto()
    elif isinstance(path_or_config, (str, Path)):
        if root_dir:
            cfg_path = root_dir.expanduser() / path_or_config
        else:
            cfg_path = Path(path_or_config).expanduser()
        if not Path(cfg_path).exists():
            raise InitializationError(
                "ConfigFileNotFoundError",
                "user",
                f"Configuration file '{cfg_path}' does not exist., current working directory is {Path.cwd()}",
            )
        cfg = config.auto(cfg_path)
    else:
        cfg = path_or_config

    logger.info(f"Flyte config initialized as {cfg}", extra={"highlighter": ReprHighlighter()})

    # parse image, this will overwrite the image_refs set in the config file
    parse_images(cfg, images)

    await init.aio(
        org=cfg.task.org,
        project=cfg.task.project,
        domain=cfg.task.domain,
        endpoint=cfg.platform.endpoint,
        insecure=cfg.platform.insecure,
        insecure_skip_verify=cfg.platform.insecure_skip_verify,
        ca_cert_file_path=cfg.platform.ca_cert_file_path,
        auth_type=cfg.platform.auth_mode,
        command=cfg.platform.command,
        proxy_command=cfg.platform.proxy_command,
        client_id=cfg.platform.client_id,
        client_credentials_secret=cfg.platform.client_credentials_secret,
        root_dir=root_dir,
        log_level=log_level,
        log_format=log_format,
        image_builder=cfg.image.builder,
        images=cfg.image.image_refs,
        storage=storage,
        source_config_path=cfg_path,
        sync_local_sys_paths=sync_local_sys_paths,
    )


@syncify
async def init_in_cluster(
    org: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    insecure: bool = False,
) -> dict[str, typing.Any]:
    import os

    from flyte._utils import str2bool

    PROJECT_NAME = "FLYTE_INTERNAL_EXECUTION_PROJECT"
    DOMAIN_NAME = "FLYTE_INTERNAL_EXECUTION_DOMAIN"
    ORG_NAME = "_U_ORG_NAME"
    ENDPOINT_OVERRIDE = "_U_EP_OVERRIDE"
    INSECURE_SKIP_VERIFY_OVERRIDE = "_U_INSECURE_SKIP_VERIFY"
    _UNION_EAGER_API_KEY_ENV_VAR = "_UNION_EAGER_API_KEY"

    org = org or os.getenv(ORG_NAME)
    project = project or os.getenv(PROJECT_NAME)
    domain = domain or os.getenv(DOMAIN_NAME)
    api_key = api_key or os.getenv(_UNION_EAGER_API_KEY_ENV_VAR)

    remote_kwargs: dict[str, typing.Any] = {"insecure": False}
    if api_key:
        logger.info("Using api key from environment")
        remote_kwargs["api_key"] = api_key
    else:
        ep = endpoint or os.environ.get(ENDPOINT_OVERRIDE, "host.docker.internal:8090")
        remote_kwargs["endpoint"] = ep
        if not insecure:
            if "localhost" in ep or "docker" in ep:
                remote_kwargs["insecure"] = True
        logger.debug(f"Using controller endpoint: {ep} with kwargs: {remote_kwargs}")

    # Check for insecure_skip_verify override (e.g. for self-signed certs)
    insecure_skip_verify_str = os.getenv(INSECURE_SKIP_VERIFY_OVERRIDE, "")
    if str2bool(insecure_skip_verify_str):
        remote_kwargs["insecure_skip_verify"] = True
        logger.info("SSL certificate verification disabled (insecure_skip_verify=True)")

    await init.aio(org=org, project=project, domain=domain, image_builder="remote", **remote_kwargs)
    return remote_kwargs


def _get_init_config() -> Optional[_InitConfig]:
    """
    Get the current initialization configuration. Thread-safe implementation.

    :return: The current InitData if initialized, None otherwise
    """
    with _init_lock:
        return _init_config


def get_init_config() -> _InitConfig:
    """
    Get the current initialization configuration. Thread-safe implementation.

    :return: The current InitData if initialized, None otherwise
    """
    cfg = _get_init_config()
    if cfg is None:
        raise InitializationError(
            "StorageNotInitializedError",
            "user",
            "Configuration has not been initialized. Call flyte.init() with a valid endpoint or",
            " api-key before using this function.",
        )
    return cfg


def get_storage() -> Storage | None:
    """
    Get the current storage configuration. Thread-safe implementation.

    :return: The current storage configuration
    """
    cfg = _get_init_config()
    if cfg is None:
        raise InitializationError(
            "StorageNotInitializedError",
            "user",
            "Configuration has not been initialized. Call flyte.init() with a valid endpoint or",
            " api-key before using this function.",
        )
    return cfg.storage


def get_client() -> ClientSet:
    """
    Get the current client. Thread-safe implementation.

    :return: The current client
    """
    cfg = _get_init_config()
    if cfg is None or cfg.client is None:
        raise InitializationError(
            "ClientNotInitializedError",
            "user",
            "Client has not been initialized. Call flyte.init() with a valid endpoint or"
            " api-key before using this function.",
        )
    return cfg.client


def is_initialized() -> bool:
    """
    Check if the system has been initialized.

    :return: True if initialized, False otherwise
    """
    return _get_init_config() is not None


def initialize_in_cluster() -> None:
    """
    Initialize the system for in-cluster execution. This is a placeholder function and does not perform any actions.

    :return: None
    """
    init()


# Define a generic type variable for the decorated function
T = TypeVar("T", bound=Callable)


def ensure_client():
    """
    Ensure that the client is initialized. If not, raise an InitializationError.
    This function is used to check if the client is initialized before executing any Flyte remote API methods.
    """
    if _get_init_config() is None or _get_init_config().client is None:
        raise InitializationError(
            "ClientNotInitializedError",
            "user",
            "Client has not been initialized. Call flyte.init() with a valid endpoint"
            " or api-key before using this function.",
        )


def requires_storage(func: T) -> T:
    """
    Decorator that checks if the storage has been initialized before executing the function.
    Raises InitializationError if the storage is not initialized.

    :param func: Function to decorate
    :return: Decorated function that checks for initialization
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _get_init_config() is None or _get_init_config().storage is None:
            raise InitializationError(
                "StorageNotInitializedError",
                "user",
                f"Function '{func.__name__}' requires storage to be initialized. "
                f"Call flyte.init() with a valid storage configuration before using this function.",
            )
        return func(*args, **kwargs)

    return typing.cast(T, wrapper)


def requires_upload_location(func: T) -> T:
    """
    Decorator that checks if the storage has been initialized before executing the function.
    Raises InitializationError if the storage is not initialized.

    :param func: Function to decorate
    :return: Decorated function that checks for initialization
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        from ._context import internal_ctx

        ctx = internal_ctx()
        if not ctx.raw_data:
            raise InitializationError(
                "No upload path configured",
                "user",
                f"Function '{func.__name__}' requires client to be initialized. "
                f"Call flyte.init() with storage configuration before using this function.",
            )
        return func(*args, **kwargs)

    return typing.cast(T, wrapper)


def requires_initialization(func: T) -> T:
    """
    Decorator that checks if the system has been initialized before executing the function.
    Raises InitializationError if the system is not initialized.

    :param func: Function to decorate
    :return: Decorated function that checks for initialization
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        if not is_initialized():
            raise InitializationError(
                "NotInitConfiguredError",
                "user",
                f"Function '{func.__name__}' requires initialization. Call flyte.init() before using this function.",
            )
        return func(*args, **kwargs)

    return typing.cast(T, wrapper)


def require_project_and_domain(func):
    """
    Decorator that ensures the current Flyte configuration defines
    both 'project' and 'domain'. Raises a clear error if not found.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cfg = get_init_config()
        if cfg.project is None:
            raise ValueError(
                "Project must be provided to initialize the client. "
                "Please set 'project' in the 'task' section of your config file, "
                "or pass it directly to flyte.init(project='your-project-name')."
            )

        if cfg.domain is None:
            raise ValueError(
                "Domain must be provided to initialize the client. "
                "Please set 'domain' in the 'task' section of your config file, "
                "or pass it directly to flyte.init(domain='your-domain-name')."
            )

        return func(*args, **kwargs)

    return wrapper


async def _init_for_testing(
    project: str | None = None,
    domain: str | None = None,
    root_dir: Path | None = None,
    log_level: int | None = None,
    client: ClientSet | None = None,
):
    from flyte._utils.helpers import get_cwd_editable_install

    global _init_config  # noqa: PLW0603

    if log_level:
        initialize_logger(log_level=log_level)

    with _init_lock:
        root_dir = root_dir or get_cwd_editable_install() or Path.cwd()
        _init_config = _InitConfig(
            root_dir=root_dir,
            project=project,
            domain=domain,
            client=client,
        )


def replace_client(client):
    global _init_config  # noqa: PLW0603

    with _init_lock:
        _init_config = _init_config.replace(client=client)


def current_domain() -> str:
    """
    Returns the current domain from Runtime environment (on the cluster) or from the initialized configuration.
    This is safe to be used during `deploy`, `run` and within `task` code.

    NOTE: This will not work if you deploy a task to a domain and then run it in another domain.

    Raises InitializationError if the configuration is not initialized or domain is not set.
    :return: The current domain
    """
    from ._context import ctx

    tctx = ctx()
    if tctx is not None:
        domain = tctx.action.domain
        if domain is not None:
            return domain

    cfg = _get_init_config()
    if cfg is None or cfg.domain is None:
        raise InitializationError(
            "DomainNotInitializedError",
            "user",
            "Domain has not been initialized. Call flyte.init() with a valid domain before using this function.",
        )
    return cfg.domain
