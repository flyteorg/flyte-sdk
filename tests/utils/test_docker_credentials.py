import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flyte._utils.docker_credentials import (
    _load_docker_config,
    create_dockerconfigjson_from_config,
    create_dockerconfigjson_from_credentials,
)


def test_create_dockerconfigjson_from_credentials():
    """Test creating dockerconfigjson from explicit credentials."""
    registry = "ghcr.io"
    username = "myuser"
    password = "mytoken"

    result = create_dockerconfigjson_from_credentials(registry, username, password)

    # Parse the result
    config = json.loads(result)

    # Verify structure
    assert "auths" in config
    assert registry in config["auths"]
    assert "auth" in config["auths"][registry]

    # Decode and verify auth token
    auth_token = config["auths"][registry]["auth"]
    decoded = base64.b64decode(auth_token).decode()
    assert decoded == f"{username}:{password}"


def test_load_docker_config(tmp_path):
    """Test loading Docker config from file."""
    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {
            "docker.io": {"auth": "dGVzdDp0ZXN0"},
            "ghcr.io": {"auth": "dXNlcjpwYXNz"},
        }
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    loaded_config = _load_docker_config(config_file)
    assert loaded_config == test_config


def test_load_docker_config_from_env(tmp_path, monkeypatch):
    """Test loading Docker config from DOCKER_CONFIG env var."""
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    config_file = docker_dir / "config.json"
    test_config = {"auths": {"docker.io": {"auth": "dGVzdDp0ZXN0"}}}

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    monkeypatch.setenv("DOCKER_CONFIG", str(docker_dir))

    loaded_config = _load_docker_config()
    assert loaded_config == test_config


def test_load_docker_config_file_not_found():
    """Test error handling when Docker config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        _load_docker_config("/nonexistent/path/config.json")


def test_create_dockerconfigjson_from_config_with_direct_auth(tmp_path):
    """Test creating dockerconfigjson from config with direct auth tokens."""
    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {
            "docker.io": {"auth": "dGVzdDp0ZXN0"},
            "ghcr.io": {"auth": "dXNlcjpwYXNz"},
        }
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    result = create_dockerconfigjson_from_config(
        registries=["ghcr.io"], docker_config_path=config_file
    )

    config = json.loads(result)
    assert "auths" in config
    assert "ghcr.io" in config["auths"]
    assert config["auths"]["ghcr.io"]["auth"] == "dXNlcjpwYXNz"
    assert "docker.io" not in config["auths"]  # Should be filtered out


def test_create_dockerconfigjson_from_config_all_registries(tmp_path):
    """Test creating dockerconfigjson with all registries when none specified."""
    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {
            "docker.io": {"auth": "dGVzdDp0ZXN0"},
            "ghcr.io": {"auth": "dXNlcjpwYXNz"},
        }
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    result = create_dockerconfigjson_from_config(docker_config_path=config_file)

    config = json.loads(result)
    assert "auths" in config
    assert "ghcr.io" in config["auths"]
    assert "docker.io" in config["auths"]


@patch("flyte._utils.docker_credentials._get_credentials_from_helper")
def test_create_dockerconfigjson_from_config_with_cred_helper(mock_get_creds, tmp_path):
    """Test creating dockerconfigjson from config using credential helper."""
    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {"ghcr.io": {}},  # No direct auth
        "credHelpers": {"ghcr.io": "osxkeychain"},
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    # Mock the credential helper to return username/password
    mock_get_creds.return_value = ("myuser", "mytoken")

    result = create_dockerconfigjson_from_config(
        registries=["ghcr.io"], docker_config_path=config_file
    )

    config = json.loads(result)
    assert "auths" in config
    assert "ghcr.io" in config["auths"]

    # Decode and verify the auth token was created from helper credentials
    auth_token = config["auths"]["ghcr.io"]["auth"]
    decoded = base64.b64decode(auth_token).decode()
    assert decoded == "myuser:mytoken"


@patch("flyte._utils.docker_credentials._get_credentials_from_helper")
def test_create_dockerconfigjson_from_config_with_global_creds_store(
    mock_get_creds, tmp_path
):
    """Test using global credsStore instead of per-registry credHelpers."""
    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {"docker.io": {}},
        "credsStore": "osxkeychain",  # Global credential store
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    mock_get_creds.return_value = ("user", "pass")

    result = create_dockerconfigjson_from_config(
        registries=["docker.io"], docker_config_path=config_file
    )

    config = json.loads(result)
    assert "docker.io" in config["auths"]


def test_create_dockerconfigjson_from_config_no_credentials(tmp_path):
    """Test error handling when no credentials can be extracted."""
    config_file = tmp_path / "config.json"
    test_config = {"auths": {"ghcr.io": {}}}  # No auth, no helpers

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    with pytest.raises(ValueError, match="No credentials could be extracted"):
        create_dockerconfigjson_from_config(
            registries=["ghcr.io"], docker_config_path=config_file
        )


def test_create_dockerconfigjson_from_config_empty_auths(tmp_path):
    """Test error handling when config has no registries."""
    config_file = tmp_path / "config.json"
    test_config = {"auths": {}}

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    with pytest.raises(ValueError, match="No registries found"):
        create_dockerconfigjson_from_config(docker_config_path=config_file)
