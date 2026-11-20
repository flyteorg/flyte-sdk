import asyncio

import pytest

fastapi = pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
from fastapi.testclient import TestClient  # noqa: E402

from flyte.app.extras import FastAPIAppEnvironment  # noqa: E402

BASE = "/onebox/apps/my-app"


def _served_app(monkeypatch, env: FastAPIAppEnvironment):
    served = {}

    async def fake_serve(self, *args, **kwargs):
        served["app"] = self.config.loaded_app if self.config.loaded else self.config.app

    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    asyncio.run(env._fastapi_app_server())
    return served["app"]


def _app() -> "fastapi.FastAPI":
    app = fastapi.FastAPI()

    @app.get("/hello")
    def hello(request: fastapi.Request):
        return {"root_path": request.scope.get("root_path"), "url": str(request.url_for("hello"))}

    return app


# A path-based app URL: requests reach the app with the path intact (the default) or
# without it (a stripping proxy); routing and generated URLs work either way.
@pytest.mark.parametrize("path", [f"{BASE}/hello", "/hello"])
def test_serves_under_app_base_path(monkeypatch, path):
    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)
    served = _served_app(monkeypatch, FastAPIAppEnvironment(name="base-path-app", app=_app()))
    resp = TestClient(served).get(path)
    assert resp.status_code == 200
    assert resp.json()["root_path"] == BASE
    assert resp.json()["url"].endswith(f"{BASE}/hello")


string_app = _app()


# uvicorn also takes the app as an import string: the loaded app is wrapped, not the string.
@pytest.mark.parametrize("path", [f"{BASE}/hello", "/hello"])
def test_serves_an_import_string_app_under_app_base_path(monkeypatch, path):
    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)
    env = FastAPIAppEnvironment(
        name="string-app", app=string_app, uvicorn_config=uvicorn.Config(f"{__name__}:string_app", port=8080)
    )
    resp = TestClient(_served_app(monkeypatch, env)).get(path)
    assert resp.status_code == 200
    assert resp.json()["root_path"] == BASE
    assert resp.json()["url"].endswith(f"{BASE}/hello")


def test_docs_point_under_base_path(monkeypatch):
    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)
    served = _served_app(monkeypatch, FastAPIAppEnvironment(name="docs-app", app=_app()))
    resp = TestClient(served).get(f"{BASE}/docs")
    assert resp.status_code == 200
    assert f"{BASE}/openapi.json" in resp.text


def test_untouched_for_host_based_urls(monkeypatch):
    monkeypatch.delenv("_U_APP_BASE_PATH", raising=False)
    app = _app()
    assert _served_app(monkeypatch, FastAPIAppEnvironment(name="host-app", app=app)) is app


def test_explicit_root_path_wins(monkeypatch):
    monkeypatch.setenv("_U_APP_BASE_PATH", BASE)
    app = _app()
    env = FastAPIAppEnvironment(
        name="explicit-app", app=app, uvicorn_config=uvicorn.Config(app, port=8080, root_path="/custom")
    )
    assert _served_app(monkeypatch, env) is app
