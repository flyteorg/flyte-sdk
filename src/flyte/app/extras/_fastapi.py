from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import rich.repr

import flyte.app
from flyte.models import SerializationContext

if TYPE_CHECKING:
    import fastapi
    import uvicorn


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

        if self.uvicorn_config is None:
            self.uvicorn_config = uvicorn.Config(self.app, port=self.port.port)
        elif self.uvicorn_config is not None:
            if self.uvicorn_config.port is None:
                self.uvicorn_config.port = self.port.port

        await uvicorn.Server(self.uvicorn_config).serve()

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []
