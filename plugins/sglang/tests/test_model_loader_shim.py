"""Unit tests for FlyteModelLoader in the sglang-fserve shim.

The shim imports torch and sglang at module scope and neither is installed in the test
environment, so both are replaced with stubs in ``sys.modules`` before the shim is imported.
The Flyte loader module is stubbed as well: it imports torch for its dtype table, and these
tests only need a ``SafeTensorsStreamer`` they can control.
"""

import importlib
import sys
import types
from unittest import mock

import pytest

SHIM_MODULE = "flyteplugins.sglang._model_loader.shim"
DEFAULT_WEIGHTS = [("model.embed_tokens.weight", "t0"), ("model.norm.weight", "t1")]


class _FakeDefaultModelLoader:
    def _get_weights_iterator(self, source):
        return iter(DEFAULT_WEIGHTS)


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def shim(monkeypatch):
    from flyteplugins.sglang._constants import SGLANG_MIN_VERSION_STR

    stubs = {
        "torch": mock.MagicMock(),
        "sglang": _module("sglang", __version__=SGLANG_MIN_VERSION_STR),
        "sglang.srt": _module("sglang.srt"),
        "sglang.srt.model_loader": _module("sglang.srt.model_loader"),
        "sglang.srt.model_loader.loader": _module(
            "sglang.srt.model_loader.loader", DefaultModelLoader=_FakeDefaultModelLoader
        ),
        "sglang.srt.configs": _module("sglang.srt.configs"),
        "sglang.srt.configs.device_config": _module("sglang.srt.configs.device_config", DeviceConfig=object),
        "sglang.srt.configs.model_config": _module("sglang.srt.configs.model_config", ModelConfig=object),
        "sglang.srt.server_args": _module("sglang.srt.server_args", prepare_server_args=mock.Mock()),
        "sglang.srt.utils": _module("sglang.srt.utils", kill_process_tree=mock.Mock()),
        "sglang.srt.server": _module("sglang.srt.server", launch_server=mock.Mock()),
        "flyte.app.extras._model_loader.loader": _module(
            "flyte.app.extras._model_loader.loader", SafeTensorsStreamer=mock.Mock(), prefetch=mock.Mock()
        ),
    }
    # Attribute access (``sglang.srt.model_loader.loader.X``) walks the parents, not sys.modules.
    stubs["sglang"].srt = stubs["sglang.srt"]
    stubs["sglang.srt"].model_loader = stubs["sglang.srt.model_loader"]
    stubs["sglang.srt.model_loader"].loader = stubs["sglang.srt.model_loader.loader"]

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, SHIM_MODULE, raising=False)

    yield importlib.import_module(SHIM_MODULE)

    sys.modules.pop(SHIM_MODULE, None)


def test_falls_back_to_default_weights_when_streamer_is_unavailable(shim, monkeypatch):
    """A ``return`` in place of ``yield from`` here silently hands SGLang zero weights."""
    monkeypatch.setattr(shim, "SafeTensorsStreamer", mock.Mock(side_effect=ValueError("no streamer")))
    source = types.SimpleNamespace(prefix="")

    weights = list(shim.FlyteModelLoader()._get_weights_iterator(source))

    assert weights == DEFAULT_WEIGHTS


def test_streams_weights_with_source_prefix(shim, monkeypatch):
    streamer = mock.Mock()
    streamer.get_tensors.return_value = iter([("layer.weight", "t0"), ("layer.bias", "t1")])
    monkeypatch.setattr(shim, "SafeTensorsStreamer", mock.Mock(return_value=streamer))
    source = types.SimpleNamespace(prefix="model.")

    weights = list(shim.FlyteModelLoader()._get_weights_iterator(source))

    assert weights == [("model.layer.weight", "t0"), ("model.layer.bias", "t1")]


BASE = "/onebox/apps/llm"


class _FakeHTTPApp:
    """SGLang's module-level FastAPI app, as far as the shim touches it."""

    def __init__(self):
        self.middleware = []

    def add_middleware(self, cls, **kwargs):
        self.middleware.append((cls, kwargs))


def _sglang_http_app(monkeypatch):
    app = _FakeHTTPApp()
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.http_server", _module("sglang.srt.entrypoints.http_server", app=app)
    )
    return app


@pytest.mark.asyncio
async def test_serves_sglang_routes_under_the_base_path(shim, monkeypatch):
    app = _sglang_http_app(monkeypatch)
    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)

    shim._serve_under_base_path()

    assert len(app.middleware) == 1
    cls, kwargs = app.middleware[0]
    for path in (BASE + "/v1/models", "/v1/models"):
        scope = await _scope_seen_by(lambda inner: cls(inner, **kwargs), path)
        assert (scope["root_path"], scope["path"]) == (BASE, BASE + "/v1/models"), path


def test_no_base_path_leaves_sglang_unchanged(shim, monkeypatch):
    app = _sglang_http_app(monkeypatch)
    monkeypatch.delenv("_U_APP_BASE_PATH", raising=False)

    shim._serve_under_base_path()

    assert app.middleware == []


def test_an_sglang_without_its_http_app_is_left_alone(shim, monkeypatch, caplog):
    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.http_server", _module("sglang.srt.entrypoints.http_server")
    )

    shim._serve_under_base_path()

    assert "does not serve under the app's base path" in caplog.text


async def _scope_seen_by(app_factory, path: str) -> dict:
    """The ASGI scope an app sees behind the middleware, for a request to path."""
    seen = {}

    async def inner(scope, receive, send):
        seen.update(scope)

    await app_factory(inner)({"type": "http", "path": path, "raw_path": path.encode(), "root_path": ""}, None, None)
    return seen
