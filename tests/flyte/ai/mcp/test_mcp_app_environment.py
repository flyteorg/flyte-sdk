"""Tests for generic :class:`MCPAppEnvironment`."""

import contextlib

import pytest

import flyte.app
from flyte.ai.mcp import MCPAppEnvironment


class TestMCPAppEnvironmentBasics:
    def test_requires_fastmcp_instance(self):
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP(name="unit-test")
        env = MCPAppEnvironment(name="generic-mcp", mcp=mcp)
        assert env.mcp is mcp
        assert env._mcp_server is mcp
        assert env.type == "MCPApp"

    def test_default_mount_and_transport(self):
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP(name="unit-test")
        env = MCPAppEnvironment(name="generic-mcp", mcp=mcp)
        assert env.mcp_mount_path == "/mcp"
        assert env.transport == "streamable-http"

    def test_starlette_mount_path(self):
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP(name="unit-test")
        env = MCPAppEnvironment(name="generic-mcp", mcp=mcp, mcp_mount_path="/custom")
        route_paths = [route.path for route in env._starlette_app.routes]
        assert "/custom" in route_paths

    def test_health_route(self):
        from mcp.server.fastmcp import FastMCP
        from starlette.testclient import TestClient

        mcp = FastMCP(name="unit-test")

        @mcp.tool()
        def ping() -> str:
            return "pong"

        env = MCPAppEnvironment(name="generic-mcp", mcp=mcp)
        client = TestClient(env._starlette_app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_relative_mount_path_rejected(self):
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP(name="unit-test")
        with pytest.raises(ValueError, match="absolute path"):
            MCPAppEnvironment(name="generic-mcp", mcp=mcp, mcp_mount_path="mcp")

    def test_inherits_app_environment(self):
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP(name="unit-test")
        env = MCPAppEnvironment(name="generic-mcp", mcp=mcp)
        assert isinstance(env, flyte.app.AppEnvironment)


class TestStdioTransport:
    """``transport="stdio"`` builds no HTTP surface and runs on stdin/stdout."""

    @staticmethod
    def _env(**kwargs):
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP(name="unit-test")

        @mcp.tool()
        def ping() -> str:
            return "pong"

        return MCPAppEnvironment(name="stdio-mcp", mcp=mcp, transport="stdio", **kwargs)

    def test_builds_no_starlette_app(self):
        assert self._env()._starlette_app is None

    def test_advertises_no_http_links(self):
        # A stdio server has no reachable URL, so surfacing /health or a mount path
        # in the console would point at nothing.
        assert self._env().links == []

    def test_leaves_server_unset_so_serve_fails_fast(self):
        # flyte.serve() would run _server on a background thread and then poll an HTTP
        # health check that can never succeed. Leaving _server unset makes serve() raise
        # immediately instead of hanging forever.
        assert self._env()._server is None

    def test_http_transports_still_build_starlette_app(self):
        from mcp.server.fastmcp import FastMCP

        for transport in ("streamable-http", "sse"):
            env = MCPAppEnvironment(name="http-mcp", mcp=FastMCP(name="unit-test"), transport=transport)
            assert env._starlette_app is not None, transport
            assert env._server is not None, transport
            assert env.links, transport

    def test_run_stdio_async_rejects_non_stdio_transport(self):
        import asyncio

        from mcp.server.fastmcp import FastMCP

        env = MCPAppEnvironment(name="http-mcp", mcp=FastMCP(name="unit-test"))
        with pytest.raises(ValueError, match="requires transport='stdio'"):
            asyncio.run(env.run_stdio_async())

    def test_run_stdio_async_serves_a_session(self):
        """Drive a real initialize/tools-list handshake over in-memory streams."""
        import asyncio
        from unittest.mock import patch

        import anyio
        import mcp.types as types
        from mcp.shared.memory import create_client_server_memory_streams

        async def exercise():
            env = self._env()
            async with create_client_server_memory_streams() as (client_streams, server_streams):
                client_read, client_write = client_streams

                # run_stdio_async() binds the process's stdin/stdout; swap in the paired
                # in-memory streams so the test drives a genuine session.
                def fake_stdio_server():
                    @contextlib.asynccontextmanager
                    async def _cm():
                        yield server_streams

                    return _cm()

                async with anyio.create_task_group() as tg:
                    with patch("mcp.server.fastmcp.server.stdio_server", fake_stdio_server):
                        tg.start_soon(env.run_stdio_async)

                        from mcp.client.session import ClientSession

                        async with ClientSession(client_read, client_write) as session:
                            init = await session.initialize()
                            assert init.serverInfo.name == "unit-test"
                            listed = await session.list_tools()
                            assert [t.name for t in listed.tools] == ["ping"]
                            called = await session.call_tool("ping", {})
                            assert isinstance(called.content[0], types.TextContent)
                            assert called.content[0].text == "pong"
                    tg.cancel_scope.cancel()

        asyncio.run(exercise())
