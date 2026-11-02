from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import rich.repr

import flyte.app
from flyte.models import SerializationContext

if TYPE_CHECKING:
    import fastapi
    import uvicorn

# The path an app with a path-based URL is served under (e.g.
# /onebox/apps/<name>), set by the platform on the app's container.
APP_BASE_PATH_ENV_VAR = "_U_APP_BASE_PATH"


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class FastAPIAppEnvironment(flyte.app.AppEnvironment):
    app: fastapi.FastAPI
    type: str = "FastAPI"
    uvicorn_config: uvicorn.Config | None = None

    def __post_init__(self):
        try:
            import fastapi
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "fastapi is not installed. Please install 'fastapi' to use FastAPIAppEnvironment."
            )

        # starlette is a dependency of fastapi, so if fastapi is installed, starlette is also installed.
        try:
            from starlette.datastructures import State
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "starlette is not installed. Please install 'starlette' to use FastAPIAppEnvironment."
            )

        class PicklableState(State):
            def __getstate__(self):
                state = self.__dict__.copy()
                # Replace the unpicklable State with an empty dict
                state["_state"] = {}
                return state

            def __setstate__(self, state):
                self.__dict__.update(state)
                # Restore a fresh State object
                self.state = State()

        # NOTE: since FastAPI cannot be pickled (because starlette.datastructures.State cannot be pickled due to
        # circular references), we need to patch the state object to make it picklable.
        self.app.state = PicklableState()

        super().__post_init__()
        if self.app is None:
            raise ValueError("app cannot be None for FastAPIAppEnvironment")
        if not isinstance(self.app, fastapi.FastAPI):
            raise TypeError(f"app must be of type fastapi.FastAPI, got {type(self.app)}")

        self.links = [flyte.app.Link(path="/docs", title="FastAPI OpenAPI Docs", is_relative=True), *self.links]
        self._server = self._fastapi_app_server

    async def _fastapi_app_server(self):
        try:
            import uvicorn
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "uvicorn is not installed. Please install 'uvicorn' to use FastAPIAppEnvironment."
            )

        port = self.get_port().port  # the host-assigned port when running natively on an Apple Silicon devbox
        if self.uvicorn_config is None:
            self.uvicorn_config = uvicorn.Config(self.app, port=port)
        elif self.uvicorn_config is not None:
            if self.uvicorn_config.port is None:
                self.uvicorn_config.port = port
        # An app served under a path of a shared host (a path-based app URL,
        # e.g. https://platform.example.com/onebox/apps/<name>/): the platform
        # tells it the path. Not uvicorn's root_path, which assumes the proxy
        # removed the path (it prepends it again); requests may arrive either way.
        base_path = os.getenv(APP_BASE_PATH_ENV_VAR, "").rstrip("/")
        if base_path and not self.uvicorn_config.root_path:
            self.uvicorn_config.app = _ServeUnderBasePath(self.uvicorn_config.app, base_path)

        await uvicorn.Server(self.uvicorn_config).serve()

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []


class _ServeUnderBasePath:
    """Serves an ASGI app under a base path, whether or not the proxy in front removed it.

    ASGI's root_path is the path the app is mounted under, and a request's path includes it;
    Starlette (and FastAPI) route on the path below root_path and build URLs (url_for, the
    docs' openapi.json) with it. A request that arrives without the base path, from a proxy
    that removed it, gets it put back, so routing and generated URLs agree either way.
    """

    def __init__(self, app, base_path: str):
        self.app = app
        self.base_path = base_path

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope = dict(scope)
            path = scope.get("path", "")
            if not (path == self.base_path or path.startswith(self.base_path + "/")):
                scope["path"] = self.base_path + path
                if scope.get("raw_path") is not None:
                    scope["raw_path"] = self.base_path.encode() + scope["raw_path"]
            scope["root_path"] = self.base_path
        await self.app(scope, receive, send)
