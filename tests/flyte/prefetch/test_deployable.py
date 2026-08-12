"""Tests for flyte.prefetch._deployable module."""

import importlib
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def deployable():
    """
    Import the module fresh.

    Its environment is built at import time from process environment variables,
    so a test that changes them has to reimport to see the effect.
    """
    sys.modules.pop("flyte.prefetch._deployable", None)
    return importlib.import_module("flyte.prefetch._deployable")


def test_registers_under_the_prefetch_environment(deployable):
    """
    The registered name is the contract: a backend addresses this task by name
    and version, so renaming it silently breaks every caller that is not Python.

    TaskEnvironment prefixes task names with the environment's name, which is why
    the bare `hf-model` that build-image gets is not available to a @env.task.
    """
    assert deployable.prefetch_env.name == "prefetch"
    assert deployable.hf_model.name == "prefetch.hf_model"


def test_no_secret_is_attached_by_default(deployable):
    """
    Naming a secret that does not exist wedges the pod in
    CreateContainerConfigError with no HuggingFace error to read -- an opaque
    failure in exactly the first-run case. Public repos need no token.
    """
    assert deployable.prefetch_env.secrets is None


def test_hf_token_key_attaches_a_secret_when_set():
    """A caller that needs a gated repo opts in through the environment."""
    sys.modules.pop("flyte.prefetch._deployable", None)
    with patch.dict("os.environ", {"FLYTE_PREFETCH_HF_TOKEN_KEY": "my-hf-token"}):
        module = importlib.import_module("flyte.prefetch._deployable")

    secrets = module.prefetch_env.secrets
    assert secrets is not None
    assert len(secrets) == 1
    assert secrets[0].key == "my-hf-token"
    assert secrets[0].as_env_var == "HF_TOKEN"

    # Leave the module cache holding the default-environment build.
    sys.modules.pop("flyte.prefetch._deployable", None)


def test_image_override_skips_the_build():
    """
    A fully-qualified image is the only practical way to deploy this against a
    locally-built SDK, since the default base installs the published flyte.
    """
    sys.modules.pop("flyte.prefetch._deployable", None)
    with patch.dict("os.environ", {"FLYTE_PREFETCH_IMAGE": "ghcr.io/acme/prefetch:v3"}):
        module = importlib.import_module("flyte.prefetch._deployable")

    assert module.image == "ghcr.io/acme/prefetch:v3"

    sys.modules.pop("flyte.prefetch._deployable", None)


def test_requests_disk_for_the_snapshot_fallback(deployable):
    """
    The unsharded path streams to object storage and never lands the weights, so
    the disk request covers the snapshot-download fallback, not the happy path.
    """
    resources = deployable.prefetch_env.resources
    assert resources.disk == "50Gi"
    # Unsharded only -- sharding needs vLLM, a CUDA toolchain and GPUs at
    # prefetch time, which belongs in its own environment.
    assert resources.gpu is None


def test_not_imported_by_the_prefetch_package():
    """
    Importing prefetch eagerly at the `flyte` level has been a problem before;
    this module is only ever imported by the deploy driver.

    Runs in a subprocess because this test session has already imported the
    module, so an in-process sys.modules check would always pass.
    """
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flyte.prefetch; sys.exit(1 if 'flyte.prefetch._deployable' in sys.modules else 0)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"importing flyte.prefetch pulled in _deployable\n{result.stderr}"


def test_empty_optional_strings_become_none(deployable):
    """
    Flat strings with "" for unset: an optional becomes a union literal that is
    awkward to build from a non-Python caller, and every input has to be sent
    explicitly anyway because CreateRun does not apply declared defaults.
    """
    captured = {}

    def fake_store(info_json, raw_data_path=None):
        import json

        captured.update(json.loads(info_json))
        return "dir"

    with patch.object(deployable, "store_hf_model_task", side_effect=fake_store):
        result = deployable.hf_model.func("Qwen/Qwen3-0.6B", "", "")

    assert result == "dir"
    assert captured["repo"] == "Qwen/Qwen3-0.6B"
    assert captured["artifact_name"] is None
    assert captured["short_description"] is None


def test_supplied_optional_strings_are_passed_through(deployable):
    """Non-empty values reach HuggingFaceModelInfo unchanged."""
    captured = {}

    def fake_store(info_json, raw_data_path=None):
        import json

        captured.update(json.loads(info_json))
        return "dir"

    with patch.object(deployable, "store_hf_model_task", side_effect=fake_store):
        deployable.hf_model.func("Qwen/Qwen3-0.6B", "my-model", "a small model")

    assert captured["artifact_name"] == "my-model"
    assert captured["short_description"] == "a small model"
