from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from flyte.cli.main import main
from flyte.remote._client.auth._public_client_cache import (
    CachedPublicClientAuthMetadata,
    get_public_client_auth_metadata_cache_path,
)


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


def test_create_secret_no_value(runner: CliRunner):
    result = runner.invoke(main, ["create", "secret", "my_secret"])
    assert result.exit_code == 1
    assert result.stdout.startswith("Enter secret value: ")


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_secret_value(mock_cli_config, mock_secret_create, runner: CliRunner):
    mock_secret_create.return_value = None

    secret_value = "my_value"

    result = runner.invoke(main, ["create", "secret", "my_secret", "--value", secret_value])
    assert result.exit_code == 0, result.stderr
    mock_secret_create.assert_called_once_with(name="my_secret", value=b"my_value", type="regular", cluster_pool=None)


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_secret_from_file(mock_cli_config, mock_secret_create, runner: CliRunner, tmp_path):
    mock_secret_create.return_value = None

    secret_value = "my_value"
    with open(tmp_path / "secret.txt", "w") as f:
        f.write(secret_value)

    result = runner.invoke(main, ["create", "secret", "my_secret", "--from-file", str(tmp_path / "secret.txt")])
    assert result.exit_code == 0, result.stderr
    mock_secret_create.assert_called_once_with(name="my_secret", value=b"my_value", type="regular", cluster_pool=None)


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_secret_with_cluster_pool(mock_cli_config, mock_secret_create, runner: CliRunner):
    mock_secret_create.return_value = None

    result = runner.invoke(main, ["create", "secret", "my_secret", "--value", "my_value", "--cluster-pool", "pool-a"])
    assert result.exit_code == 0, result.stderr
    mock_secret_create.assert_called_once_with(
        name="my_secret", value=b"my_value", type="regular", cluster_pool="pool-a"
    )


def test_create_secret_cluster_pool_rejects_project(runner: CliRunner):
    result = runner.invoke(
        main,
        ["create", "secret", "my_secret", "--value", "v", "--cluster-pool", "pool-a", "--project", "p"],
    )
    assert result.exit_code == 2
    assert "Illegal usage" in result.stderr
    assert "cluster_pool" in result.stderr
    assert "project" in result.stderr


def test_create_secret_cluster_pool_rejects_domain(runner: CliRunner):
    result = runner.invoke(
        main,
        ["create", "secret", "my_secret", "--value", "v", "--cluster-pool", "pool-a", "--domain", "d"],
    )
    assert result.exit_code == 2
    assert "Illegal usage" in result.stderr
    assert "cluster_pool" in result.stderr
    assert "domain" in result.stderr


def test_create_secret_invalid_combination(runner: CliRunner):
    result = runner.invoke(main, ["create", "secret", "my_secret", "--value", "my_value", "--from-file", "my_file"])
    assert result.exit_code == 2
    # The error message includes all mutually exclusive options
    assert "Illegal usage" in result.stderr
    assert "are mutually exclusive" in result.stderr


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_interactive(mock_cli_config, mock_secret_create, runner: CliRunner):
    """Test creating image pull secret with interactive prompts."""
    mock_secret_create.return_value = None

    result = runner.invoke(
        main,
        ["create", "secret", "my_secret", "--type", "image_pull"],
        input="ghcr.io\nmyuser\nmytoken\n",
    )

    assert result.exit_code == 0, result.stderr
    assert mock_secret_create.called
    call_args = mock_secret_create.call_args
    assert call_args[1]["name"] == "my_secret"
    assert call_args[1]["type"] == "image_pull"

    # Verify the value is valid dockerconfigjson
    import json

    value = call_args[1]["value"]
    config = json.loads(value)
    assert "auths" in config
    assert "ghcr.io" in config["auths"]


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_explicit_credentials(mock_cli_config, mock_secret_create, runner: CliRunner):
    """Test creating image pull secret with explicit credentials."""
    mock_secret_create.return_value = None

    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--registry",
            "docker.io",
            "--username",
            "testuser",
            "--password",
            "testpass",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert mock_secret_create.called
    call_args = mock_secret_create.call_args
    assert call_args[1]["name"] == "my_secret"
    assert call_args[1]["type"] == "image_pull"

    # Verify the value is valid dockerconfigjson
    import base64
    import json

    value = call_args[1]["value"]
    config = json.loads(value)
    assert "auths" in config
    assert "docker.io" in config["auths"]

    # Verify credentials
    auth_token = config["auths"]["docker.io"]["auth"]
    decoded = base64.b64decode(auth_token).decode()
    assert decoded == "testuser:testpass"


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_explicit_credentials_prompt_password(
    mock_cli_config, mock_secret_create, runner: CliRunner
):
    """Test creating image pull secret with registry and username, prompting for password."""
    mock_secret_create.return_value = None

    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--registry",
            "ghcr.io",
            "--username",
            "user",
        ],
        input="mypassword\n",
    )

    assert result.exit_code == 0, result.stderr
    assert mock_secret_create.called


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_from_docker_config(mock_cli_config, mock_secret_create, runner: CliRunner, tmp_path):
    """Test creating image pull secret from Docker config file."""
    mock_secret_create.return_value = None

    # Create a test Docker config
    import json

    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {
            "docker.io": {"auth": "dGVzdDp0ZXN0"},
            "ghcr.io": {"auth": "dXNlcjpwYXNz"},
        }
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--from-docker-config",
            "--docker-config-path",
            str(config_file),
            "--registries",
            "ghcr.io",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert mock_secret_create.called
    call_args = mock_secret_create.call_args
    assert call_args[1]["name"] == "my_secret"
    assert call_args[1]["type"] == "image_pull"

    # Verify the value contains only the specified registry
    value = call_args[1]["value"]
    config = json.loads(value)
    assert "auths" in config
    assert "ghcr.io" in config["auths"]
    assert "docker.io" not in config["auths"]


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_from_docker_config_all_registries(
    mock_cli_config, mock_secret_create, runner: CliRunner, tmp_path
):
    """Test creating image pull secret from Docker config with all registries."""
    mock_secret_create.return_value = None

    import json

    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {
            "docker.io": {"auth": "dGVzdDp0ZXN0"},
            "ghcr.io": {"auth": "dXNlcjpwYXNz"},
        }
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--from-docker-config",
            "--docker-config-path",
            str(config_file),
        ],
    )

    assert result.exit_code == 0, result.stderr
    call_args = mock_secret_create.call_args
    value = call_args[1]["value"]
    config = json.loads(value)
    assert "docker.io" in config["auths"]
    assert "ghcr.io" in config["auths"]


def test_create_image_pull_secret_invalid_combination_registry_and_from_docker_config(runner: CliRunner):
    """Test that --registry and --from-docker-config are mutually exclusive."""
    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--registry",
            "ghcr.io",
            "--from-docker-config",
        ],
    )
    assert result.exit_code == 2
    error_msg = "are mutually exclusive"
    assert error_msg in result.stderr


def test_config_with_params_preserves_local():
    """Verify Config.with_params() doesn't drop LocalConfig."""
    from flyte.config._config import Config, LocalConfig, PlatformConfig, TaskConfig

    cfg = Config(local=LocalConfig(persistence=True), source=None)
    updated = cfg.with_params(PlatformConfig(), TaskConfig())
    assert updated.local.persistence is True


def test_create_config_local_persistence_only(runner: CliRunner, tmp_path):
    """Test that --local-persistence alone (no endpoint/org) succeeds."""
    outpath = str(tmp_path / "config.yaml")
    result = runner.invoke(
        main,
        ["create", "config", "--local-persistence", "-o", outpath, "--force"],
    )
    assert result.exit_code == 0, result.output
    with open(outpath) as f:
        d = yaml.safe_load(f)
    assert d["local"]["persistence"] is True
    assert "admin" not in d
    assert "task" not in d


def test_create_config_no_flags_fails(runner: CliRunner, tmp_path):
    """Test that no flags at all still raises an error."""
    outpath = str(tmp_path / "config.yaml")
    result = runner.invoke(
        main,
        ["create", "config", "-o", outpath, "--force"],
    )
    assert result.exit_code != 0
    assert "--local-persistence" in result.output


def test_create_config_requires_domain_when_endpoint_is_provided(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = str(tmp_path / "config.yaml")
    result = runner.invoke(
        main,
        [
            "create",
            "config",
            "--endpoint",
            "dns:///dogfood.cloud-staging.union.ai",
            "--org",
            "dogfood",
            "-o",
            outpath,
            "--force",
        ],
    )
    assert result.exit_code != 0
    assert "--domain must be provided" in result.output


def test_create_config_with_local_persistence(runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Test that --local-persistence writes the local.persistence field to the config YAML."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = str(tmp_path / "config.yaml")
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="test-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
    )

    with patch("flyte.cli._create.fetch_public_client_auth_metadata_sync", return_value=metadata):
        result = runner.invoke(
            main,
            [
                "create",
                "config",
                "--endpoint",
                "dns:///test.example.com",
                "--domain",
                "development",
                "--local-persistence",
                "-o",
                outpath,
                "--force",
            ],
        )

    assert result.exit_code == 0, result.output
    with open(outpath) as f:
        d = yaml.safe_load(f)
    assert "local" in d
    assert d["local"]["persistence"] is True


def test_create_config_without_local_persistence(runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Test that without --local-persistence the local section is omitted."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = str(tmp_path / "config.yaml")
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="test-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
    )

    with patch("flyte.cli._create.fetch_public_client_auth_metadata_sync", return_value=metadata):
        result = runner.invoke(
            main,
            [
                "create",
                "config",
                "--endpoint",
                "dns:///test.example.com",
                "--domain",
                "development",
                "-o",
                outpath,
                "--force",
            ],
        )

    assert result.exit_code == 0, result.output
    with open(outpath) as f:
        d = yaml.safe_load(f)
    assert d.get("local") is None


def test_create_config_fetches_and_caches_auth_metadata_when_cache_is_missing(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = str(tmp_path / "config.yaml")
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="dogfood-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
    )

    with patch("flyte.cli._create.fetch_public_client_auth_metadata_sync", return_value=metadata) as fetch_metadata:
        result = runner.invoke(
            main,
            [
                "create",
                "config",
                "--endpoint",
                "dns:///dogfood.cloud-staging.union.ai",
                "--org",
                "dogfood",
                "--project",
                "edward-test",
                "--domain",
                "staging",
                "--image-builder",
                "remote",
                "-o",
                outpath,
                "--force",
            ],
        )

    assert result.exit_code == 0, result.output
    fetch_metadata.assert_called_once()
    with open(outpath) as f:
        d = yaml.safe_load(f)
    assert d["admin"] == {"endpoint": "dns:///dogfood.cloud-staging.union.ai"}
    cache_path = get_public_client_auth_metadata_cache_path("dogfood", "staging")
    assert cache_path.exists()
    with cache_path.open() as f:
        cached = yaml.safe_load(f)
    assert cached == {
        "authType": "Pkce",
        "clientId": "dogfood-uctl",
        "insecure": False,
        "authorizationHeader": "flyte-authorization",
        "redirectUri": "http://localhost:53593/callback",
        "scopes": ["all"],
    }


def test_create_config_aborts_when_cached_auth_metadata_exists_and_user_declines_overwrite(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = str(tmp_path / "config.yaml")
    cache_path = get_public_client_auth_metadata_cache_path("dogfood", "staging")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("existing: true")

    with patch("flyte.cli._create.fetch_public_client_auth_metadata_sync") as fetch_metadata:
        result = runner.invoke(
            main,
            [
                "create",
                "config",
                "--endpoint",
                "dns:///dogfood.cloud-staging.union.ai",
                "--org",
                "dogfood",
                "--domain",
                "staging",
                "-o",
                outpath,
            ],
            input="n\n",
        )

    assert result.exit_code == 0, result.output
    assert "Will not overwrite the existing auth metadata cache" in result.output
    fetch_metadata.assert_not_called()
    assert not Path(outpath).exists()


def test_create_config_refreshes_cached_auth_metadata_when_user_accepts_overwrite(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = str(tmp_path / "config.yaml")
    cache_path = get_public_client_auth_metadata_cache_path("dogfood", "staging")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("existing: true")
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="dogfood-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
    )

    with patch("flyte.cli._create.fetch_public_client_auth_metadata_sync", return_value=metadata) as fetch_metadata:
        result = runner.invoke(
            main,
            [
                "create",
                "config",
                "--endpoint",
                "dns:///dogfood.cloud-staging.union.ai",
                "--org",
                "dogfood",
                "--domain",
                "staging",
                "-o",
                outpath,
                "--force",
            ],
        )

    assert result.exit_code == 0, result.output
    fetch_metadata.assert_called_once()
    with open(outpath) as f:
        d = yaml.safe_load(f)
    assert d["admin"] == {"endpoint": "dns:///dogfood.cloud-staging.union.ai"}


def test_create_config_prompts_for_config_file_overwrite_after_cache_resolution(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = Path(tmp_path / "config.yaml")
    outpath.write_text("original")
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="dogfood-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
    )

    with patch("flyte.cli._create.fetch_public_client_auth_metadata_sync", return_value=metadata):
        result = runner.invoke(
            main,
            [
                "create",
                "config",
                "--endpoint",
                "dns:///dogfood.cloud-staging.union.ai",
                "--org",
                "dogfood",
                "--domain",
                "staging",
                "-o",
                str(outpath),
            ],
            input="n\n",
        )

    assert result.exit_code == 0, result.output
    assert outpath.read_text() == "original"


def test_create_config_fetch_failure_aborts_before_writing_cache_or_config(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = Path(tmp_path / "config.yaml")

    with patch("flyte.cli._create.fetch_public_client_auth_metadata_sync", side_effect=RuntimeError("boom")):
        result = runner.invoke(
            main,
            [
                "create",
                "config",
                "--endpoint",
                "dns:///dogfood.cloud-staging.union.ai",
                "--org",
                "dogfood",
                "--domain",
                "staging",
                "-o",
                str(outpath),
                "--force",
            ],
        )

    assert result.exit_code != 0
    assert not outpath.exists()
    assert not get_public_client_auth_metadata_cache_path("dogfood", "staging").exists()


def test_create_config_cache_write_failure_does_not_block_config_write(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outpath = Path(tmp_path / "config.yaml")
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="dogfood-gcp-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
    )

    with (
        patch("flyte.cli._create.fetch_public_client_auth_metadata_sync", return_value=metadata),
        patch("flyte.cli._create.write_cached_public_client_auth_metadata", return_value=None),
    ):
        result = runner.invoke(
            main,
            [
                "create",
                "config",
                "--endpoint",
                "dns:///dogfood-gcp.cloud-staging.union.ai",
                "--project",
                "edward-test",
                "--domain",
                "staging",
                "-o",
                str(outpath),
                "--force",
            ],
        )

    assert result.exit_code == 0, result.output
    with outpath.open() as f:
        d = yaml.safe_load(f)
    assert d["task"]["org"] == "dogfood-gcp"
    assert d["admin"] == {"endpoint": "dns:///dogfood-gcp.cloud-staging.union.ai"}
