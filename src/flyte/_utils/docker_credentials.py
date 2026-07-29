"""Helper functions for creating Docker registry credentials for image pull secrets."""

import base64
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_JSON = "config.json"
_DEFAULT_CONFIG_PATH = f"~/.docker/{_CONFIG_JSON}"
_CRED_HELPERS = "credHelpers"
_CREDS_STORE = "credsStore"

# Docker Hub is stored under a handful of aliases in ~/.docker/config.json; all normalize to
# the pushable "docker.io" registry host.
_DOCKER_HUB_REGISTRY = "docker.io"
_DOCKER_HUB_KEYS = frozenset(
    {
        "https://index.docker.io/v1/",
        "https://index.docker.io/v2/",
        "index.docker.io",
        "registry-1.docker.io",
        "docker.io",
    }
)


def _normalize_registry(server: str) -> str:
    """Normalize a Docker config ``auths`` key to a registry hostname usable as a push target.

    Docker Hub is written as a full URL (``https://index.docker.io/v1/``); every Hub alias
    collapses to ``docker.io``. Other registries are stored as bare hosts and pass through.
    """
    host = server
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split("/", 1)[0]
    if server in _DOCKER_HUB_KEYS or host in _DOCKER_HUB_KEYS or host.endswith(".docker.io") or host == "docker.io":
        return _DOCKER_HUB_REGISTRY
    return host


def infer_registry_from_docker_config(docker_config_path: str | Path | None = None) -> str | None:
    """Best-effort inference of a push registry from the user's Docker config.

    Enumerates the ``auths`` entries (present even when credentials live in a ``credsStore`` —
    the entries exist, they're just empty), normalizes Docker Hub's various keys to
    ``docker.io``, and — for Hub — resolves the namespace (``docker.io/<username>``) via the
    configured credential helper. If Hub is absent and exactly one other registry is present,
    that registry is returned instead. Hub is preferred when both are present.

    Returns ``None`` when nothing can be inferred (missing config, no auths, helper
    unavailable). This never raises: inference is advisory and must never block a command.
    """
    try:
        config = _load_docker_config(docker_config_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    auths = config.get("auths", {})
    if not auths:
        return None

    hub_keys: list[str] = []
    non_hub: list[str] = []
    for server in auths:
        if _normalize_registry(server) == _DOCKER_HUB_REGISTRY:
            hub_keys.append(server)
        else:
            non_hub.append(_normalize_registry(server))

    # Prefer Docker Hub: resolve the namespace so the candidate is docker.io/<username>.
    if hub_keys:
        server = hub_keys[0]
        helper = _get_credential_helper(config, server)
        if helper:
            creds = _get_credentials_from_helper(helper, server)
            if creds and creds[0]:
                return f"{_DOCKER_HUB_REGISTRY}/{creds[0]}"
        # Fall back to a directly-stored auth token's username if present.
        auth = auths.get(server, {}).get("auth")
        if auth:
            try:
                username = base64.b64decode(auth).decode().split(":", 1)[0]
                if username:
                    return f"{_DOCKER_HUB_REGISTRY}/{username}"
            except Exception:
                pass
        # Hub login present but namespace unknown — a bare "docker.io" is not a valid push
        # target, so decline rather than propose something that will fail.
        return None

    # No Hub login: if exactly one other registry is configured, use it.
    unique = list(dict.fromkeys(non_hub))
    if len(unique) == 1:
        return unique[0]
    return None


def _load_docker_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load Docker config from specified path.

    Args:
        config_path: Path to Docker config file. If None, uses DOCKER_CONFIG env var
                    or defaults to ~/.docker/config.json

    Returns:
        Dictionary containing Docker config

    Raises:
        FileNotFoundError: If the config file does not exist
        json.JSONDecodeError: If the config file is not valid JSON
    """
    if not config_path:
        docker_config_env = os.environ.get("DOCKER_CONFIG")
        if docker_config_env:
            config_path = Path(docker_config_env) / _CONFIG_JSON
        else:
            config_path = Path(_DEFAULT_CONFIG_PATH).expanduser()
    else:
        config_path = Path(config_path).expanduser()

    with open(config_path) as f:
        return json.load(f)


def _get_credential_helper(config: dict[str, Any], registry: str | None = None) -> str | None:
    """Get credential helper for registry or global default."""
    if registry and _CRED_HELPERS in config and registry in config[_CRED_HELPERS]:
        return config[_CRED_HELPERS].get(registry)
    return config.get(_CREDS_STORE)


def _get_credentials_from_helper(helper: str, registry: str) -> tuple[str, str] | None:
    """
    Get credentials from system credential helper.

    Args:
        helper: Name of the credential helper (e.g., "osxkeychain", "wincred")
        registry: Registry hostname to get credentials for

    Returns:
        Tuple of (username, password) or None if credentials cannot be retrieved
    """
    helper_cmd = f"docker-credential-{helper}"

    try:
        process = subprocess.Popen(
            [helper_cmd, "get"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, error = process.communicate(input=registry)

        if process.returncode != 0:
            logger.error(f"Credential helper error: {error}")
            return None

        creds = json.loads(output)
        return creds.get("Username"), creds.get("Secret")
    except FileNotFoundError:
        logger.error(f"Credential helper {helper_cmd} not found in PATH")
        return None
    except Exception as e:
        logger.error(f"Error getting credentials: {e!s}")
        return None


def create_dockerconfigjson_from_config(
    registries: list[str] | None = None,
    docker_config_path: str | Path | None = None,
) -> str:
    """
    Create a dockerconfigjson string from existing Docker config.

    This function extracts Docker registry credentials from the user's Docker config file
    and creates a JSON string containing only the credentials for the specified registries.
    It handles credentials stored directly in the config file as well as those managed by
    credential helpers.

    Args:
        registries: List of registries to extract credentials for. If None, all registries
                   from the config will be used.
        docker_config_path: Path to the Docker config file. If None, the function will look
                          for the config file in the standard locations.

    Returns:
        JSON string in dockerconfigjson format: {"auths": {"registry": {"auth": "..."}}}

    Raises:
        FileNotFoundError: If Docker config file cannot be found
        ValueError: If no credentials can be extracted
    """
    config = _load_docker_config(docker_config_path)

    # Create new config structure with empty auths
    new_config: dict[str, Any] = {"auths": {}}

    # Use specified registries or all from config
    target_registries = registries or list(config.get("auths", {}).keys())

    if not target_registries:
        raise ValueError("No registries found in Docker config and none specified")

    for registry in target_registries:
        registry_config = config.get("auths", {}).get(registry, {})
        if registry_config.get("auth"):
            # Direct auth token exists
            new_config["auths"][registry] = {"auth": registry_config["auth"]}
        else:
            # Try to get credentials from helper
            helper = _get_credential_helper(config, registry)
            if helper:
                creds = _get_credentials_from_helper(helper, registry)
                if creds:
                    username, password = creds
                    auth_string = f"{username}:{password}"
                    new_config["auths"][registry] = {"auth": base64.b64encode(auth_string.encode()).decode()}
                else:
                    logger.warning(f"Could not retrieve credentials for {registry} from credential helper")
            else:
                logger.warning(f"No credentials found for {registry}")

    if not new_config["auths"]:
        raise ValueError(f"No credentials could be extracted for registries: {', '.join(target_registries)}")

    return json.dumps(new_config)


def create_dockerconfigjson_from_credentials(
    registry: str,
    username: str,
    password: str,
) -> str:
    """
    Create a dockerconfigjson string from explicit credentials.

    Args:
        registry: Registry hostname (e.g., "ghcr.io", "docker.io")
        username: Username or token name for the registry
        password: Password or access token for the registry

    Returns:
        JSON string in dockerconfigjson format: {"auths": {"registry": {"auth": "..."}}}
    """
    auth_string = f"{username}:{password}"
    auth_token = base64.b64encode(auth_string.encode()).decode()

    config = {"auths": {registry: {"auth": auth_token}}}

    return json.dumps(config)
