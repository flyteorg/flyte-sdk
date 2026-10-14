"""Tests for the context-scoped init-config override (``flyte._initialize.init_config_context``)."""

import asyncio
from pathlib import Path

import pytest

import flyte._initialize as init_mod
from flyte._initialize import _get_init_config, _InitConfig, init_config_context


def _cfg(org: str) -> _InitConfig:
    return _InitConfig(root_dir=Path("/tmp"), org=org)


@pytest.fixture
def global_cfg(monkeypatch):
    """Install a known module-global config for the duration of a test."""
    cfg = _cfg("global-org")
    monkeypatch.setattr(init_mod, "_init_config", cfg)
    return cfg


def test_unset_context_falls_back_to_global(global_cfg):
    assert _get_init_config() is global_cfg


def test_context_override_wins_and_is_restored(global_cfg):
    override = _cfg("tenant-a")
    with init_config_context(override):
        assert _get_init_config() is override
    assert _get_init_config() is global_cfg


def test_override_applies_with_no_global(monkeypatch):
    monkeypatch.setattr(init_mod, "_init_config", None)
    override = _cfg("tenant-a")
    with init_config_context(override):
        assert _get_init_config() is override
    assert _get_init_config() is None


def test_nested_overrides(global_cfg):
    outer, inner = _cfg("outer"), _cfg("inner")
    with init_config_context(outer):
        with init_config_context(inner):
            assert _get_init_config() is inner
        assert _get_init_config() is outer
    assert _get_init_config() is global_cfg


def test_override_restored_on_exception(global_cfg):
    with pytest.raises(RuntimeError):
        with init_config_context(_cfg("tenant-a")):
            raise RuntimeError("boom")
    assert _get_init_config() is global_cfg


@pytest.mark.asyncio
async def test_concurrent_tasks_are_isolated(global_cfg):
    """Two concurrent tasks with different configs must each observe their own."""
    started = asyncio.Event()

    async def worker(org: str, *, wait_for_other: bool) -> list:
        cfg = _cfg(org)
        seen = []
        with init_config_context(cfg):
            seen.append(_get_init_config())
            if wait_for_other:
                started.set()
            else:
                await started.wait()
            # Yield so the two tasks interleave inside their respective contexts.
            await asyncio.sleep(0)
            seen.append(_get_init_config())
        seen.append(_get_init_config())
        return seen

    a, b = await asyncio.gather(worker("tenant-a", wait_for_other=True), worker("tenant-b", wait_for_other=False))

    assert [c.org for c in a[:2]] == ["tenant-a", "tenant-a"]
    assert [c.org for c in b[:2]] == ["tenant-b", "tenant-b"]
    # After each task exits its block, the global is visible again.
    assert a[2] is global_cfg
    assert b[2] is global_cfg
    # And the parent task was never affected.
    assert _get_init_config() is global_cfg


@pytest.mark.asyncio
async def test_child_task_inherits_override(global_cfg):
    override = _cfg("tenant-a")

    async def child():
        return _get_init_config()

    with init_config_context(override):
        # Tasks copy the current context at creation time, which is what the ASGI
        # middleware relies on when it hands the request off downstream.
        assert await asyncio.create_task(child()) is override


def test_get_client_uses_override(monkeypatch):
    from flyte._initialize import get_client

    monkeypatch.setattr(init_mod, "_init_config", None)
    sentinel = object()
    with init_config_context(_InitConfig(root_dir=Path("/tmp"), client=sentinel)):
        assert get_client() is sentinel


def test_current_project_uses_override(monkeypatch):
    from flyte._initialize import current_project

    monkeypatch.setattr(init_mod, "_init_config", _cfg("global-org"))
    with init_config_context(_InitConfig(root_dir=Path("/tmp"), project="ctx-project")):
        assert current_project() == "ctx-project"
