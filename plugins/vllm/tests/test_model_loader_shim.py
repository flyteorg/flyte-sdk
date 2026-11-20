"""Unit tests for the vllm-fserve shim's base path handling.

The shim imports torch and vllm at module scope and neither is installed in the test
environment, so both are replaced with stubs in ``sys.modules`` before the shim is imported.
"""

import importlib
import sys
import types
from unittest import mock

import pytest

SHIM_MODULE = "flyteplugins.vllm._model_loader.shim"
BASE = "/onebox/apps/llm"


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def shim(monkeypatch):
    from flyteplugins.vllm._constants import VLLM_MIN_VERSION_STR

    class _DefaultModelLoader:
        Source = object

    stubs = {
        "torch": mock.MagicMock(),
        "vllm": _module("vllm", __version__=VLLM_MIN_VERSION_STR),
        "vllm.entrypoints": _module("vllm.entrypoints"),
        "vllm.entrypoints.cli": _module("vllm.entrypoints.cli"),
        "vllm.entrypoints.cli.main": _module("vllm.entrypoints.cli.main", main=mock.Mock()),
        "vllm.config": _module("vllm.config", ModelConfig=object, VllmConfig=object),
        "vllm.distributed": _module("vllm.distributed", get_tensor_model_parallel_rank=mock.Mock()),
        "vllm.model_executor": _module("vllm.model_executor"),
        "vllm.model_executor.model_loader": _module(
            "vllm.model_executor.model_loader", register_model_loader=mock.Mock()
        ),
        "vllm.model_executor.model_loader.default_loader": _module(
            "vllm.model_executor.model_loader.default_loader", DefaultModelLoader=_DefaultModelLoader
        ),
        "vllm.model_executor.model_loader.dummy_loader": _module(
            "vllm.model_executor.model_loader.dummy_loader", DummyModelLoader=object
        ),
        "vllm.model_executor.model_loader.sharded_state_loader": _module(
            "vllm.model_executor.model_loader.sharded_state_loader", ShardedStateLoader=object
        ),
        "vllm.model_executor.model_loader.utils": _module(
            "vllm.model_executor.model_loader.utils", set_default_torch_dtype=mock.Mock()
        ),
        "flyte.app.extras._model_loader.loader": _module(
            "flyte.app.extras._model_loader.loader", SafeTensorsStreamer=mock.Mock(), prefetch=mock.Mock()
        ),
    }
    stubs["vllm"].entrypoints = stubs["vllm.entrypoints"]
    stubs["vllm.entrypoints"].cli = stubs["vllm.entrypoints.cli"]
    stubs["vllm.entrypoints.cli"].main = stubs["vllm.entrypoints.cli.main"]
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, SHIM_MODULE, raising=False)

    yield importlib.import_module(SHIM_MODULE)

    sys.modules.pop(SHIM_MODULE, None)


def test_serve_gets_the_base_path_middleware(shim, monkeypatch):
    from flyte.app._base_path import SERVE_UNDER_BASE_PATH_MIDDLEWARE

    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)
    argv = ["vllm-fserve", "serve", "/model", "--port", "8080"]
    shim._serve_under_base_path(argv)
    assert argv[-2:] == ["--middleware", SERVE_UNDER_BASE_PATH_MIDDLEWARE]

    # Once only.
    shim._serve_under_base_path(argv)
    assert argv.count(SERVE_UNDER_BASE_PATH_MIDDLEWARE) == 1


def test_no_base_path_or_another_subcommand_changes_nothing(shim, monkeypatch):
    monkeypatch.delenv("_U_APP_BASE_PATH", raising=False)
    argv = ["vllm-fserve", "serve", "/model"]
    shim._serve_under_base_path(argv)
    assert argv == ["vllm-fserve", "serve", "/model"]

    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)
    argv = ["vllm-fserve", "bench", "serve"]
    shim._serve_under_base_path(argv)
    assert argv == ["vllm-fserve", "bench", "serve"]


@pytest.mark.asyncio
async def test_the_middleware_serves_vllm_routes_under_the_base_path(shim, monkeypatch):
    """Constructed the way vLLM installs a --middleware class (entrypoints/serve/middleware/register.py):
    imported by name, then `cls(app)`. Routes see the path below root_path, prefixed or not."""
    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)
    argv = ["vllm-fserve", "serve", "/model"]
    shim._serve_under_base_path(argv)
    module_path, object_name = argv[-1].rsplit(".", 1)
    middleware = getattr(importlib.import_module(module_path), object_name)

    for path in (BASE + "/v1/models", "/v1/models"):
        scope = await _scope_seen_by(middleware, path)
        assert (scope["root_path"], scope["path"]) == (BASE, BASE + "/v1/models"), path


async def _scope_seen_by(app_factory, path: str) -> dict:
    """The ASGI scope an app sees behind the middleware, for a request to path."""
    seen = {}

    async def inner(scope, receive, send):
        seen.update(scope)

    await app_factory(inner)({"type": "http", "path": path, "raw_path": path.encode(), "root_path": ""}, None, None)
    return seen
