"""Tests for `flyte.clustered.jax_initialize` (jax is stubbed in sys.modules; never imported for real)."""

from __future__ import annotations

import sys
import types
from unittest.mock import Mock

import pytest

from flyte.clustered import jax_initialize

TOPOLOGY = {
    "RANK": "2",
    "WORLD_SIZE": "4",
    "MASTER_ADDR": "f-abc123-workers-0-0.f-abc123.my-project-development.svc.cluster.local",
    "MASTER_PORT": "29500",
}


@pytest.fixture
def fake_jax(monkeypatch):
    jax = types.ModuleType("jax")
    distributed = types.ModuleType("jax.distributed")
    distributed.initialize = Mock()  # type: ignore[attr-defined]
    distributed.is_initialized = Mock(return_value=False)  # type: ignore[attr-defined]
    jax.distributed = distributed  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.distributed", distributed)
    return jax


def _set_topology(monkeypatch, env=TOPOLOGY):
    for k in TOPOLOGY:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_jax_initialize_passes_cluster_params(monkeypatch, fake_jax):
    _set_topology(monkeypatch)

    jax_initialize()

    fake_jax.distributed.initialize.assert_called_once_with(
        coordinator_address=f"{TOPOLOGY['MASTER_ADDR']}:29500",
        num_processes=4,
        process_id=2,
        cluster_detection_method="deactivate",
    )


def test_jax_initialize_overrides_win(monkeypatch, fake_jax):
    _set_topology(monkeypatch)

    jax_initialize(local_device_ids=[0], initialization_timeout=60, process_id=7)

    kwargs = fake_jax.distributed.initialize.call_args.kwargs
    assert kwargs["local_device_ids"] == [0]
    assert kwargs["initialization_timeout"] == 60
    assert kwargs["process_id"] == 7
    assert kwargs["num_processes"] == 4
    assert kwargs["cluster_detection_method"] == "deactivate"


def test_jax_initialize_outside_cluster_raises(monkeypatch, fake_jax):
    _set_topology(monkeypatch, {})

    with pytest.raises(RuntimeError, match="JaxRun"):
        jax_initialize()

    fake_jax.distributed.initialize.assert_not_called()


def test_jax_initialize_idempotent(monkeypatch, fake_jax):
    _set_topology(monkeypatch)
    fake_jax.distributed.is_initialized.return_value = True

    jax_initialize()

    fake_jax.distributed.initialize.assert_not_called()


def test_jax_initialize_without_is_initialized(monkeypatch, fake_jax):
    """Older JAX releases have no jax.distributed.is_initialized; initialize is still called."""
    _set_topology(monkeypatch)
    del fake_jax.distributed.is_initialized

    jax_initialize()

    fake_jax.distributed.initialize.assert_called_once()
