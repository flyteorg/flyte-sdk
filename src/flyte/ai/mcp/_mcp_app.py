from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import rich.repr

import flyte.app
from flyte._image import Image
from flyte._resources import Resources
from flyte.app._types import Link
from flyte.models import SerializationContext

if TYPE_CHECKING:
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from starlette.applications import Starlette


@dataclass(kw_only=True, repr=True)
class MCPAppEnvironment(flyte.app.AppEnvironment):
    """Serve a FastMCP server over HTTP (Starlette + Uvicorn).

    Pass a configured ``FastMCP`` instance and optional HTTP layout settings.
    Install extras with ``pip install 'flyte[mcp]'``.

    **HTTP layout**

    - ``GET /health`` — liveness/readiness JSON ``{"status": "healthy"}``.
    - The MCP ASGI app is mounted at ``mcp_mount_path`` (default ``/mcp``). With
      ``transport="streamable-http"``, the session endpoint is ``{mcp_mount_path}/mcp``.
      SSE transport uses ``{mcp_mount_path}/sse`` instead.
    """

    type: str = "MCPApp"

    mcp: FastMCP

    mcp_mount_path: str = "/mcp"
    transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http"
    uvicorn_config: uvicorn.Config | None = None

    _starlette_app: Starlette | None = field(init=False, default=None)
    _caller_frame: inspect.FrameInfo | None = field(init=False, default=None)

    def __post_init__(self):
        if getattr(self, "image", None) in (None, "auto"):
            self.image = Image.from_debian_base().with_pip_packages("mcp", "starlette", "uvicorn")
        if getattr(self, "resources", None) is None:
            self.resources = Resources(cpu=1, memory="512Mi")

        super().__post_init__()

        if self.transport not in ["stdio", "sse", "streamable-http"]:
            raise ValueError("transport must be either 'stdio', 'sse', or 'streamable-http'.")

        if not isinstance(self.mcp_mount_path, str) or not self.mcp_mount_path.startswith("/"):
            raise ValueError("mcp_mount_path must be an absolute path starting with '/'.")

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            if caller_frame and caller_frame.f_back:
                self._caller_frame = inspect.getframeinfo(caller_frame.f_back)

        self._starlette_app = self._create_starlette_app()

        mcp_link = self.mcp_mount_path
        if self.transport == "streamable-http":
            mcp_link = f"{self.mcp_mount_path}/mcp"
        elif self.transport == "sse":
            mcp_link = f"{self.mcp_mount_path}/sse"

        self.links = [
            Link(path=mcp_link, title="MCP Endpoint", is_relative=True),
            Link(path="/health", title="Health", is_relative=True),
            *self.links,
        ]

        self._server = self._starlette_app_server

    @property
    def _mcp_server(self) -> FastMCP:
        """Alias for :attr:`mcp` (matches historical attribute name)."""
        return self.mcp

    def _create_starlette_app(self) -> Starlette:
        try:
            from starlette.applications import Starlette
            from starlette.responses import JSONResponse
            from starlette.routing import Mount, Route
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "starlette is not installed. Please install 'flyte[mcp]' to use MCPAppEnvironment."
            ) from e

        async def _health(_: Any) -> JSONResponse:
            return JSONResponse({"status": "healthy"})

        mcp_asgi = None
        if self.transport == "sse":
            if hasattr(self.mcp, "sse_app"):
                mcp_asgi = self.mcp.sse_app()
            else:  # pragma: no cover
                raise RuntimeError("FastMCP does not expose sse_app(); cannot use transport='sse'.")
        elif hasattr(self.mcp, "streamable_http_app"):
            mcp_asgi = self.mcp.streamable_http_app()
        elif hasattr(self.mcp, "sse_app"):  # pragma: no cover
            mcp_asgi = self.mcp.sse_app()
        else:  # pragma: no cover
            raise RuntimeError("FastMCP does not expose an ASGI app (expected streamable_http_app or sse_app).")

        routes = [
            Mount(self.mcp_mount_path, app=mcp_asgi),
            Route("/health", endpoint=_health, methods=["GET"]),
        ]

        @asynccontextmanager
        async def lifespan(_app: Starlette):
            async with mcp_asgi.router.lifespan_context(mcp_asgi):
                yield

        return Starlette(routes=routes, lifespan=lifespan)

    async def _starlette_app_server(self):
        try:
            import uvicorn
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError("uvicorn is not installed. Please install 'flyte[mcp]' to serve this app.") from e

        assert self._starlette_app is not None
        if self.uvicorn_config is None:
            self.uvicorn_config = uvicorn.Config(self._starlette_app, port=self.port.port)
        elif self.uvicorn_config.port is None:
            self.uvicorn_config.port = self.port.port

        await uvicorn.Server(self.uvicorn_config).serve()

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.name
        yield "type", self.type
        yield "mcp_mount_path", self.mcp_mount_path
