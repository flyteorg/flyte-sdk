"""`flyte.use_profile` scopes config to a block without touching global state."""

from __future__ import annotations

import pathlib
import threading

import pytest

import flyte
from flyte._initialize import _get_init_config

_CONFIG = """
task:
  org: shared-org
  project: shared
  domain: development
profiles:
  prod:
    task:
      domain: production
  gpu:
    task:
      project: gpu-proj
"""


@pytest.fixture
def cfg_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "config.yaml"
    p.write_text(_CONFIG)
    return p


def test_scopes_project_and_domain(cfg_file: pathlib.Path) -> None:
    with flyte.use_profile("prod", config_file=cfg_file) as cfg:
        assert cfg.domain == "production"
        assert cfg.project == "shared"  # inherited from the top level
        assert _get_init_config() is cfg


def test_restores_on_exit(cfg_file: pathlib.Path) -> None:
    before = _get_init_config()
    with flyte.use_profile("prod", config_file=cfg_file):
        pass
    assert _get_init_config() is before


def test_restores_on_exception(cfg_file: pathlib.Path) -> None:
    before = _get_init_config()
    with pytest.raises(RuntimeError):
        with flyte.use_profile("prod", config_file=cfg_file):
            raise RuntimeError("boom")
    assert _get_init_config() is before


def test_nests(cfg_file: pathlib.Path) -> None:
    with flyte.use_profile("prod", config_file=cfg_file):
        assert _get_init_config().domain == "production"
        with flyte.use_profile("gpu", config_file=cfg_file):
            assert _get_init_config().project == "gpu-proj"
            assert _get_init_config().domain == "development"
        assert _get_init_config().domain == "production"


def test_overrides_win_over_profile(cfg_file: pathlib.Path) -> None:
    with flyte.use_profile("prod", config_file=cfg_file, project="explicit", domain="dev2") as cfg:
        assert cfg.project == "explicit"
        assert cfg.domain == "dev2"


def test_unknown_profile_raises(cfg_file: pathlib.Path) -> None:
    with pytest.raises(flyte.config.ProfileNotFoundError):
        with flyte.use_profile("nope", config_file=cfg_file):
            pass


def test_does_not_leak_to_other_threads(cfg_file: pathlib.Path) -> None:
    """The override is context-scoped: a concurrent submitter must not see it.

    This is what makes it safe for a router to switch profiles per run while other work is in
    flight -- the module-global config is never mutated.
    """
    ambient = _get_init_config()
    seen = []
    barrier = threading.Barrier(2)

    def other() -> None:
        barrier.wait()
        seen.append(_get_init_config())
        barrier.wait()

    t = threading.Thread(target=other)
    t.start()
    with flyte.use_profile("prod", config_file=cfg_file):
        barrier.wait()
        barrier.wait()
    t.join()
    # The other thread saw whatever was ambient, never the block's override.
    assert seen == [ambient]


@pytest.mark.asyncio
async def test_async_form(cfg_file: pathlib.Path) -> None:
    before = _get_init_config()
    async with flyte.aio_use_profile("gpu", config_file=cfg_file) as cfg:
        assert cfg.project == "gpu-proj"
    assert _get_init_config() is before


def test_override_reaches_syncified_calls(cfg_file: pathlib.Path) -> None:
    """The documented use: a plain `flyte.run(...)` inside a `use_profile` block.

    `flyte.run` is syncified, so the coroutine executes on a background event-loop thread while
    the `with` block holds the override in the *calling* thread. That only works because
    `run_coroutine_threadsafe` copies the caller's context across — non-obvious enough that it
    needs asserting: if syncify ever scheduled work without copying context, this public API
    would silently submit to the wrong control plane while every other test here still passed.
    """
    from flyte.syncify import syncify

    @syncify
    async def observed_project():
        cfg = _get_init_config()
        return cfg.project if cfg else None

    with flyte.use_profile("gpu", config_file=cfg_file):
        assert observed_project() == "gpu-proj"


def test_syncified_calls_see_the_ambient_config_outside_the_block(cfg_file: pathlib.Path) -> None:
    """The other half: the override must not leak into syncified work after the block exits."""
    from flyte.syncify import syncify

    @syncify
    async def observed_project():
        cfg = _get_init_config()
        return cfg.project if cfg else None

    with flyte.use_profile("gpu", config_file=cfg_file):
        pass
    ambient = _get_init_config()
    assert observed_project() == (ambient.project if ambient else None)
