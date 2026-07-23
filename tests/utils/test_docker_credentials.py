import base64
import json
from unittest.mock import patch

import pytest

from flyte._utils.docker_credentials import (
    _load_docker_config,
    _normalize_registry,
    create_dockerconfigjson_from_config,
    create_dockerconfigjson_from_credentials,
    infer_registry_from_docker_config,
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

    result = create_dockerconfigjson_from_config(registries=["ghcr.io"], docker_config_path=config_file)

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

    result = create_dockerconfigjson_from_config(registries=["ghcr.io"], docker_config_path=config_file)

    config = json.loads(result)
    assert "auths" in config
    assert "ghcr.io" in config["auths"]

    # Decode and verify the auth token was created from helper credentials
    auth_token = config["auths"]["ghcr.io"]["auth"]
    decoded = base64.b64decode(auth_token).decode()
    assert decoded == "myuser:mytoken"


@patch("flyte._utils.docker_credentials._get_credentials_from_helper")
def test_create_dockerconfigjson_from_config_with_global_creds_store(mock_get_creds, tmp_path):
    """Test using global credsStore instead of per-registry credHelpers."""
    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {"docker.io": {}},
        "credsStore": "osxkeychain",  # Global credential store
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    mock_get_creds.return_value = ("user", "pass")

    result = create_dockerconfigjson_from_config(registries=["docker.io"], docker_config_path=config_file)

    config = json.loads(result)
    assert "docker.io" in config["auths"]


def test_create_dockerconfigjson_from_config_no_credentials(tmp_path):
    """Test error handling when no credentials can be extracted."""
    config_file = tmp_path / "config.json"
    test_config = {"auths": {"ghcr.io": {}}}  # No auth, no helpers

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    with pytest.raises(ValueError, match="No credentials could be extracted"):
        create_dockerconfigjson_from_config(registries=["ghcr.io"], docker_config_path=config_file)


def test_create_dockerconfigjson_from_config_empty_auths(tmp_path):
    """Test error handling when config has no registries."""
    config_file = tmp_path / "config.json"
    test_config = {"auths": {}}

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    with pytest.raises(ValueError, match="No registries found"):
        create_dockerconfigjson_from_config(docker_config_path=config_file)


# ---------------------------------------------------------------------------
# Push-registry inference from Docker config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server,expected",
    [
        ("https://index.docker.io/v1/", "docker.io"),
        ("index.docker.io", "docker.io"),
        ("registry-1.docker.io", "docker.io"),
        ("docker.io", "docker.io"),
        ("ghcr.io", "ghcr.io"),
        ("https://ghcr.io", "ghcr.io"),
        ("us-central1-docker.pkg.dev", "us-central1-docker.pkg.dev"),
        ("localhost:30000", "localhost:30000"),
    ],
)
def test_normalize_registry(server, expected):
    """Docker Hub's various keys collapse to docker.io; other hosts pass through."""
    assert _normalize_registry(server) == expected


def _write_config(tmp_path, config):
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    return config_file


def test_infer_registry_hub_via_cred_helper(tmp_path):
    """A Docker Hub login stored in a credsStore (empty auths entry) resolves to docker.io/<user>."""
    config_file = _write_config(
        tmp_path,
        {"auths": {"https://index.docker.io/v1/": {}}, "credsStore": "osxkeychain"},
    )
    with patch(
        "flyte._utils.docker_credentials._get_credentials_from_helper",
        return_value=("chris", "sometoken"),
    ) as mock_helper:
        result = infer_registry_from_docker_config(docker_config_path=config_file)
    assert result == "docker.io/chris"
    # The credential helper is queried with the original Hub auth key.
    mock_helper.assert_called_once_with("osxkeychain", "https://index.docker.io/v1/")


def test_infer_registry_credsstore_empty_auths_enumerated(tmp_path):
    """With a credsStore, auths entries exist but are empty — enumeration must still find Hub."""
    config_file = _write_config(
        tmp_path,
        {"auths": {"https://index.docker.io/v1/": {}}, "credsStore": "desktop"},
    )
    with patch(
        "flyte._utils.docker_credentials._get_credentials_from_helper",
        return_value=("alice", "pw"),
    ):
        assert infer_registry_from_docker_config(docker_config_path=config_file) == "docker.io/alice"


def test_infer_registry_hub_via_direct_auth(tmp_path):
    """When credentials are stored inline (no helper), the namespace comes from the auth token."""
    auth = base64.b64encode(b"bob:token").decode()
    config_file = _write_config(tmp_path, {"auths": {"https://index.docker.io/v1/": {"auth": auth}}})
    assert infer_registry_from_docker_config(docker_config_path=config_file) == "docker.io/bob"


def test_infer_registry_hub_namespace_unknown_returns_none(tmp_path):
    """Hub login present but namespace unresolvable → decline (a bare docker.io can't be pushed to)."""
    config_file = _write_config(
        tmp_path,
        {"auths": {"https://index.docker.io/v1/": {}}, "credsStore": "osxkeychain"},
    )
    with patch("flyte._utils.docker_credentials._get_credentials_from_helper", return_value=None):
        assert infer_registry_from_docker_config(docker_config_path=config_file) is None


def test_infer_registry_single_non_hub(tmp_path):
    """Exactly one non-Hub registry and no Hub login → that registry is the candidate."""
    config_file = _write_config(tmp_path, {"auths": {"ghcr.io": {"auth": "eA=="}}})
    assert infer_registry_from_docker_config(docker_config_path=config_file) == "ghcr.io"


def test_infer_registry_prefers_hub_over_non_hub(tmp_path):
    """When both Hub and another registry are present, Hub wins."""
    config_file = _write_config(
        tmp_path,
        {"auths": {"ghcr.io": {"auth": "eA=="}, "https://index.docker.io/v1/": {}}, "credsStore": "osxkeychain"},
    )
    with patch("flyte._utils.docker_credentials._get_credentials_from_helper", return_value=("carol", "pw")):
        assert infer_registry_from_docker_config(docker_config_path=config_file) == "docker.io/carol"


def test_infer_registry_multiple_non_hub_ambiguous_returns_none(tmp_path):
    """Multiple non-Hub registries with no Hub login is ambiguous → decline."""
    config_file = _write_config(tmp_path, {"auths": {"ghcr.io": {"auth": "eA=="}, "quay.io": {"auth": "eA=="}}})
    assert infer_registry_from_docker_config(docker_config_path=config_file) is None


def test_infer_registry_empty_auths_returns_none(tmp_path):
    config_file = _write_config(tmp_path, {"auths": {}})
    assert infer_registry_from_docker_config(docker_config_path=config_file) is None


def test_infer_registry_missing_config_returns_none():
    """A missing Docker config must never raise — inference is advisory."""
    assert infer_registry_from_docker_config(docker_config_path="/nonexistent/config.json") is None
