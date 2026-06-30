"""Tests for generic :class:`MCPAppEnvironment`."""

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
