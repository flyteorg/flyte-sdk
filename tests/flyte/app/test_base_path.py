import importlib

import pytest

from flyte.app._base_path import (
    GRADIO_ROOT_PATH_ENV_VAR,
    SERVE_UNDER_BASE_PATH_MIDDLEWARE,
    STREAMLIT_BASE_URL_PATH_ENV_VAR,
    ServeUnderBasePath,
    app_base_path,
    apply_base_path_defaults,
)

BASE = "/onebox/apps/my-app"


def test_app_base_path():
    assert app_base_path({"_U_APP_BASE_PATH": BASE}) == BASE
    assert app_base_path({"_U_APP_BASE_PATH": BASE + "/"}) == BASE
    assert app_base_path({}) == ""


def test_streamlit_serves_under_the_app_base_path():
    env = {"_U_APP_BASE_PATH": BASE + "/"}
    apply_base_path_defaults(env)
    assert env[STREAMLIT_BASE_URL_PATH_ENV_VAR] == BASE


def test_an_app_setting_its_own_streamlit_base_path_keeps_it():
    env = {"_U_APP_BASE_PATH": BASE, STREAMLIT_BASE_URL_PATH_ENV_VAR: "/custom"}
    apply_base_path_defaults(env)
    assert env[STREAMLIT_BASE_URL_PATH_ENV_VAR] == "/custom"


def test_apps_at_their_host_root_are_unchanged():
    env = {}
    apply_base_path_defaults(env)
    assert env == {}


def test_gradio_serves_under_the_app_base_path():
    env = {"_U_APP_BASE_PATH": BASE}
    apply_base_path_defaults(env)
    assert env[GRADIO_ROOT_PATH_ENV_VAR] == BASE


def test_an_app_setting_its_own_gradio_root_path_keeps_it():
    env = {"_U_APP_BASE_PATH": BASE, GRADIO_ROOT_PATH_ENV_VAR: "/custom"}
    apply_base_path_defaults(env)
    assert env[GRADIO_ROOT_PATH_ENV_VAR] == "/custom"


def _api():
    fastapi = pytest.importorskip("fastapi")
    app = fastapi.FastAPI()

    @app.get("/v1/models")
    def models(request: fastapi.Request):
        return {"root_path": request.scope.get("root_path", "")}

    return app


def test_middleware_serves_under_the_base_path_removed_or_not(monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("_U_APP_BASE_PATH", BASE + "/")
    app = _api()
    # As servers that take middleware by import path install it (vLLM's --middleware).
    app.add_middleware(ServeUnderBasePath)
    client = testclient.TestClient(app)
    for path in (BASE + "/v1/models", "/v1/models"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert resp.json() == {"root_path": BASE}


def test_middleware_without_a_base_path_changes_nothing(monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.delenv("_U_APP_BASE_PATH", raising=False)
    app = _api()
    app.add_middleware(ServeUnderBasePath)
    client = testclient.TestClient(app)
    assert client.get("/v1/models").json() == {"root_path": ""}
    assert client.get(BASE + "/v1/models").status_code == 404


def test_middleware_import_path_resolves():
    module, name = SERVE_UNDER_BASE_PATH_MIDDLEWARE.rsplit(".", 1)
    assert getattr(importlib.import_module(module), name) is ServeUnderBasePath
