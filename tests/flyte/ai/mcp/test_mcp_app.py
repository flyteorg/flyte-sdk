"""
Unit tests for FlyteMCPAppEnvironment.

These tests verify the FlyteMCPAppEnvironment functionality including:
- Basic instantiation and default values
- Custom configuration options
- MCP server creation with correct tools
- Starlette app creation with middleware and routes
- Tool group and tool filtering
- Allowlist helper functions
- Container command generation
- Rich repr output
"""

import pathlib

import pytest
from flyte.ai.mcp._mcp_app import (
    ALL_MCP_TOOL_GROUPS,
    ALL_MCP_TOOLS,
    TOOL_GROUP_MAPPING,
    UV_SCRIPT_EXAMPLE,
    UV_SCRIPT_FORMAT,
    _is_app_allowed,
    _is_task_allowed,
    _is_trigger_allowed,
    _resolve_tools,
    _search_files,
)

from flyte._image import Image
from flyte._resources import Resources
from flyte.ai.mcp import FlyteMCPAppEnvironment
from flyte.app._types import Domain, Scaling
from flyte.models import SerializationContext


class TestFlyteMCPAppEnvironmentInstantiation:
    """Tests for basic instantiation and default values."""

    def test_basic_instantiation_with_name_only(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.name == "test-mcp"
        assert env.type == "FlyteMCPApp"
        assert env.title is None
        assert env.description is None
        assert env.instructions is None

    def test_default_image_is_set(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.image is not None
        assert isinstance(env.image, Image)

    def test_custom_title(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", title="My MCP Server")
        assert env.title == "My MCP Server"

    def test_custom_instructions(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            instructions="Use these tools to interact with Flyte.",
        )
        assert env.instructions == "Use these tools to interact with Flyte."

    def test_custom_resources(self):
        resources = Resources(cpu=2, memory="1Gi", gpu=1)
        env = FlyteMCPAppEnvironment(name="test-mcp", resources=resources)
        assert env.resources.cpu == 2
        assert env.resources.memory == "1Gi"
        assert env.resources.gpu == 1

    def test_custom_scaling(self):
        scaling = Scaling(replicas=(2, 5), metric=Scaling.Concurrency(val=10))
        env = FlyteMCPAppEnvironment(name="test-mcp", scaling=scaling)
        assert env.scaling.replicas == (2, 5)
        assert isinstance(env.scaling.metric, Scaling.Concurrency)

    def test_custom_domain(self):
        domain = Domain(subdomain="my-mcp-subdomain")
        env = FlyteMCPAppEnvironment(name="test-mcp", domain=domain)
        assert env.domain.subdomain == "my-mcp-subdomain"

    def test_requires_auth_default_true(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.requires_auth is True

    def test_requires_auth_can_be_disabled(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", requires_auth=False)
        assert env.requires_auth is False

    def test_custom_image(self):
        custom_image = Image.from_base("python:3.12-slim")
        env = FlyteMCPAppEnvironment(name="test-mcp", image=custom_image)
        assert env.image == custom_image

    def test_custom_mcp_mount_path(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", mcp_mount_path="/sdk/mcp")
        assert env.mcp_mount_path == "/sdk/mcp"

    def test_default_mcp_mount_path(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.mcp_mount_path == "/mcp"


class TestFlyteMCPAppEnvironmentMCPServer:
    """Tests for MCP server creation."""

    def test_mcp_server_is_created(self):
        from mcp.server.fastmcp import FastMCP

        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env._mcp_server is not None
        assert isinstance(env._mcp_server, FastMCP)

    def test_mcp_server_has_tools_registered(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        assert len(tool_names) > 0
        assert "run_task" in tool_names
        assert "get_run" in tool_names

    def test_mcp_server_title_uses_name_when_no_title(self):
        env = FlyteMCPAppEnvironment(name="my-mcp")
        assert "my-mcp" in env._mcp_server.name

    def test_mcp_server_title_uses_custom_title(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", title="Custom MCP")
        assert env._mcp_server.name == "Custom MCP"

    def test_all_tools_registered_by_default(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        for tool_name in ALL_MCP_TOOLS:
            assert tool_name in tool_names, f"Tool {tool_name} not registered"


class TestFlyteMCPAppEnvironmentStarletteApp:
    """Tests for the Starlette app creation and routes."""

    def test_starlette_app_is_created(self):
        from starlette.applications import Starlette

        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env._starlette_app is not None
        assert isinstance(env._starlette_app, Starlette)

    def test_starlette_app_has_health_route(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        route_paths = [route.path for route in env._starlette_app.routes]
        assert "/health" in route_paths

    def test_starlette_app_has_mcp_mount(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        route_paths = [route.path for route in env._starlette_app.routes]
        assert "/mcp" in route_paths

    def test_starlette_app_custom_mount_path(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", mcp_mount_path="/sdk/mcp")
        route_paths = [route.path for route in env._starlette_app.routes]
        assert "/sdk/mcp" in route_paths

    def test_starlette_app_has_links(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        link_paths = [link.path for link in env.links]
        assert "/mcp" in link_paths
        assert "/health" in link_paths


class TestFlyteMCPAppEnvironmentInheritance:
    """Tests for inheritance from AppEnvironment."""

    def test_inherits_from_app_environment(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert isinstance(env, FlyteMCPAppEnvironment)

    def test_has_uvicorn_config_attribute(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.uvicorn_config is None


class TestFlyteMCPAppEnvironmentContainerCommand:
    """Tests for container command generation."""

    def test_container_command_returns_empty_list(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        ctx = SerializationContext(
            org="test-org",
            project="test-project",
            domain="test-domain",
            version="v1.0.0",
            root_dir=pathlib.Path.cwd(),
        )
        cmd = env.container_command(ctx)
        assert cmd == []


class TestFlyteMCPAppEnvironmentRichRepr:
    """Tests for rich repr output."""

    def test_rich_repr_includes_name(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        repr_items = list(env.__rich_repr__())
        names = [item[0] for item in repr_items]
        assert "name" in names

    def test_rich_repr_includes_type(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        repr_items = list(env.__rich_repr__())
        names = [item[0] for item in repr_items]
        assert "type" in names

    def test_rich_repr_includes_mcp_mount_path(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        repr_items = list(env.__rich_repr__())
        names = [item[0] for item in repr_items]
        assert "mcp_mount_path" in names

    def test_rich_repr_values(self):
        env = FlyteMCPAppEnvironment(name="my-mcp", title="My Title")
        repr_dict = dict(env.__rich_repr__())
        assert repr_dict["name"] == "my-mcp"
        assert repr_dict["title"] == "My Title"
        assert repr_dict["type"] == "FlyteMCPApp"

    def test_rich_repr_truncates_long_instructions(self):
        long_instructions = "A" * 100
        env = FlyteMCPAppEnvironment(name="test-mcp", instructions=long_instructions)
        repr_dict = dict(env.__rich_repr__())
        assert repr_dict["instructions"].endswith("...")
        assert len(repr_dict["instructions"]) < 100


class TestFlyteMCPAppEnvironmentNameValidation:
    """Tests for name validation."""

    def test_valid_names_accepted(self):
        valid_names = ["my-mcp", "mcp123", "a-b-c", "test"]
        for name in valid_names:
            env = FlyteMCPAppEnvironment(name=name)
            assert env.name == name
            env._validate_name()

    def test_invalid_names_rejected(self):
        invalid_names = ["My-MCP", "mcp_123", "-mcp", "mcp-"]
        for name in invalid_names:
            with pytest.raises(ValueError, match="must consist of lower case"):
                FlyteMCPAppEnvironment(name=name)


class TestFlyteMCPAppEnvironmentPortHandling:
    """Tests for port handling."""

    def test_default_port(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.port.port == 8080

    def test_custom_port(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", port=9000)
        assert env.port.port == 9000

    def test_reserved_ports_rejected(self):
        reserved_ports = [8012, 8022, 8112, 9090, 9091]
        for port in reserved_ports:
            with pytest.raises(ValueError, match=r"is not allowed|is reserved"):
                FlyteMCPAppEnvironment(name="test-mcp", port=port)


class TestToolFiltering:
    """Tests for tool filtering functionality."""

    def test_all_tools_enabled_by_default(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.enabled_tools == set(ALL_MCP_TOOLS)

    def test_tool_groups_single_group(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["task"])
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        assert "run_task" in tool_names
        assert "get_task" in tool_names
        assert "list_tasks" in tool_names
        assert "get_run" not in tool_names
        assert "get_app" not in tool_names

    def test_tool_groups_multiple_groups(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["task", "run"])
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        assert "run_task" in tool_names
        assert "get_run" in tool_names
        assert "wait_for_run" in tool_names
        assert "get_app" not in tool_names

    def test_individual_tools_filtering(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            tools=["run_task", "get_run"],
        )
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        assert tool_names == {"run_task", "get_run"}

    def test_cannot_specify_both_tools_and_tool_groups(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            FlyteMCPAppEnvironment(
                name="test-mcp",
                tool_groups=["task"],
                tools=["run_task"],
            )

    def test_enabled_tools_property_with_groups(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["task"])
        assert env.enabled_tools == {"run_task", "get_task", "list_tasks"}

    def test_enabled_tools_property_with_individual_tools(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tools=["run_task", "abort_run"])
        assert env.enabled_tools == {"run_task", "abort_run"}


class TestResolveTools:
    """Tests for the _resolve_tools helper function."""

    def test_resolve_tools_none_returns_all(self):
        result = _resolve_tools(None, None)
        assert result == set(ALL_MCP_TOOLS)

    def test_resolve_tools_single_group(self):
        result = _resolve_tools(["task"], None)
        assert result == {"run_task", "get_task", "list_tasks"}

    def test_resolve_tools_multiple_groups(self):
        result = _resolve_tools(["task", "run"], None)
        expected = {
            "run_task",
            "get_task",
            "list_tasks",
            "get_run",
            "get_run_io",
            "abort_run",
            "list_runs",
            "wait_for_run",
        }
        assert result == expected

    def test_resolve_tools_all_group(self):
        result = _resolve_tools(["all"], None)
        assert result == set(ALL_MCP_TOOLS)

    def test_resolve_tools_individual_tools(self):
        result = _resolve_tools(None, ["run_task", "get_run"])
        assert result == {"run_task", "get_run"}

    def test_resolve_tools_core_group_is_empty(self):
        result = _resolve_tools(["core"], None)
        assert result == set()


class TestAllowlistFunctions:
    """Tests for allowlist helper functions."""

    def test_is_task_allowed_none_allows_all(self):
        assert _is_task_allowed(None, "prod", "proj", "task") is True

    def test_is_task_allowed_full_path_match(self):
        allowlist = ["prod/proj/task"]
        assert _is_task_allowed(allowlist, "prod", "proj", "task") is True
        assert _is_task_allowed(allowlist, "dev", "proj", "task") is False

    def test_is_task_allowed_project_name_match(self):
        allowlist = ["proj/task"]
        assert _is_task_allowed(allowlist, "prod", "proj", "task") is True
        assert _is_task_allowed(allowlist, "dev", "proj", "task") is True
        assert _is_task_allowed(allowlist, "prod", "other", "task") is False

    def test_is_task_allowed_name_only_match(self):
        allowlist = ["task"]
        assert _is_task_allowed(allowlist, "prod", "proj", "task") is True
        assert _is_task_allowed(allowlist, "dev", "other", "task") is True
        assert _is_task_allowed(allowlist, "prod", "proj", "other") is False

    def test_is_app_allowed_none_allows_all(self):
        assert _is_app_allowed(None, "my-app") is True

    def test_is_app_allowed_explicit_list(self):
        allowlist = ["app-1", "app-2"]
        assert _is_app_allowed(allowlist, "app-1") is True
        assert _is_app_allowed(allowlist, "app-3") is False

    def test_is_trigger_allowed_none_allows_all(self):
        assert _is_trigger_allowed(None, "task", "trigger") is True

    def test_is_trigger_allowed_full_path_match(self):
        allowlist = ["task/trigger"]
        assert _is_trigger_allowed(allowlist, "task", "trigger") is True
        assert _is_trigger_allowed(allowlist, "other", "trigger") is False

    def test_is_trigger_allowed_name_only_match(self):
        allowlist = ["trigger"]
        assert _is_trigger_allowed(allowlist, "task1", "trigger") is True
        assert _is_trigger_allowed(allowlist, "task2", "trigger") is True
        assert _is_trigger_allowed(allowlist, "task1", "other") is False


class TestToolGroupMappingConsistency:
    """Tests for tool group mapping consistency."""

    def test_all_tools_in_at_least_one_group(self):
        tools_in_groups: set[str] = set()
        for group, tool_tuple in TOOL_GROUP_MAPPING.items():
            if group != "all":
                tools_in_groups.update(tool_tuple)
        for tool in ALL_MCP_TOOLS:
            assert tool in tools_in_groups, f"Tool {tool} not in any group"

    def test_all_group_contains_all_tools(self):
        assert set(TOOL_GROUP_MAPPING["all"]) == set(ALL_MCP_TOOLS)

    def test_all_tool_groups_exist_in_mapping(self):
        for group in ALL_MCP_TOOL_GROUPS:
            assert group in TOOL_GROUP_MAPPING, f"Group {group} not in TOOL_GROUP_MAPPING"


class TestHealthEndpoint:
    """Tests for the health check endpoint using TestClient."""

    def test_health_endpoint_returns_healthy(self):
        from starlette.testclient import TestClient

        env = FlyteMCPAppEnvironment(name="test-mcp")
        client = TestClient(env._starlette_app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestFlyteMCPAppEnvironmentAllowlists:
    """Tests for allowlist configuration on the environment."""

    def test_task_allowlist_defaults_to_none(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.task_allowlist is None

    def test_app_allowlist_defaults_to_none(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.app_allowlist is None

    def test_trigger_allowlist_defaults_to_none(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.trigger_allowlist is None

    def test_custom_task_allowlist(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            task_allowlist=["prod/proj/task1", "task2"],
        )
        assert env.task_allowlist == ["prod/proj/task1", "task2"]

    def test_custom_app_allowlist(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            app_allowlist=["app1", "app2"],
        )
        assert env.app_allowlist == ["app1", "app2"]

    def test_custom_trigger_allowlist(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            trigger_allowlist=["task/trigger1", "trigger2"],
        )
        assert env.trigger_allowlist == ["task/trigger1", "trigger2"]


class TestScriptToolGroup:
    """Tests for the script tool group."""

    def test_script_group_registers_tools(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["script"])
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        assert "build_uv_script_image_remote" in tool_names
        assert "run_uv_script_remote" in tool_names
        assert "flyte_uv_script_format" in tool_names
        assert "flyte_uv_script_example" in tool_names
        assert "run_task" not in tool_names

    def test_resolve_tools_script_group(self):
        result = _resolve_tools(["script"], None)
        expected = {
            "build_uv_script_image_remote",
            "run_uv_script_remote",
            "flyte_uv_script_format",
            "flyte_uv_script_example",
        }
        assert result == expected

    def test_individual_script_tools(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            tools=["flyte_uv_script_format", "flyte_uv_script_example"],
        )
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        assert tool_names == {"flyte_uv_script_format", "flyte_uv_script_example"}


class TestSearchToolGroup:
    """Tests for the search tool group."""

    def test_search_group_registers_tools(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["search"])
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        assert "search_flyte_sdk_examples" in tool_names
        assert "search_flyte_docs_examples" in tool_names
        assert "search_full_docs" in tool_names
        assert "run_task" not in tool_names

    def test_resolve_tools_search_group(self):
        result = _resolve_tools(["search"], None)
        expected = {
            "search_flyte_sdk_examples",
            "search_flyte_docs_examples",
            "search_full_docs",
        }
        assert result == expected

    def test_search_paths_default_to_none(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.sdk_examples_path is None
        assert env.docs_examples_path is None
        assert env.full_docs_path is None

    def test_custom_search_paths(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            sdk_examples_path="/data/sdk-examples",
            docs_examples_path="/data/docs-examples",
            full_docs_path="/data/full-docs.txt",
        )
        assert env.sdk_examples_path == "/data/sdk-examples"
        assert env.docs_examples_path == "/data/docs-examples"
        assert env.full_docs_path == "/data/full-docs.txt"


class TestUVScriptTemplates:
    """Tests for the UV script template constants."""

    def test_uv_script_format_contains_key_sections(self):
        assert "# /// script" in UV_SCRIPT_FORMAT
        assert "flyte.TaskEnvironment" in UV_SCRIPT_FORMAT
        assert "flyte.Image.from_uv_script" in UV_SCRIPT_FORMAT
        assert "@env.task" in UV_SCRIPT_FORMAT
        assert "--build" in UV_SCRIPT_FORMAT
        assert "flyte.init_passthrough" in UV_SCRIPT_FORMAT

    def test_uv_script_example_contains_key_sections(self):
        assert "# /// script" in UV_SCRIPT_EXAMPLE
        assert "scikit-learn" in UV_SCRIPT_EXAMPLE
        assert "flyte.TaskEnvironment" in UV_SCRIPT_EXAMPLE
        assert "@env.task" in UV_SCRIPT_EXAMPLE
        assert "async def main" in UV_SCRIPT_EXAMPLE
        assert "--build" in UV_SCRIPT_EXAMPLE

    def test_uv_script_format_is_stripped(self):
        assert not UV_SCRIPT_FORMAT.startswith("\n")
        assert not UV_SCRIPT_FORMAT.endswith("\n")

    def test_uv_script_example_is_stripped(self):
        assert not UV_SCRIPT_EXAMPLE.startswith("\n")
        assert not UV_SCRIPT_EXAMPLE.endswith("\n")


class TestSearchFilesHelper:
    """Tests for the _search_files async helper."""

    @pytest.mark.asyncio
    async def test_search_files_with_matches(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("import flyte\nprint('hello')\nimport flyte.io\n")

        result = await _search_files("flyte", str(tmp_path), top_n=3)
        assert "test.py" in result
        assert "flyte" in result

    @pytest.mark.asyncio
    async def test_search_files_no_matches(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello world')\n")

        result = await _search_files("nonexistent_pattern_xyz", str(tmp_path))
        assert "No matches found" in result

    @pytest.mark.asyncio
    async def test_search_files_nonexistent_path(self):
        result = await _search_files("pattern", "/nonexistent/path/abc123")
        assert "Error" in result or "No matches" in result


class TestScriptAndSearchGroupsCombined:
    """Tests for combining script and search groups with other groups."""

    def test_all_group_includes_new_tools(self):
        result = _resolve_tools(["all"], None)
        assert "build_uv_script_image_remote" in result
        assert "run_uv_script_remote" in result
        assert "flyte_uv_script_format" in result
        assert "flyte_uv_script_example" in result
        assert "search_flyte_sdk_examples" in result
        assert "search_flyte_docs_examples" in result
        assert "search_full_docs" in result

    def test_combined_script_and_task_groups(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["task", "script"])
        tool_manager = env._mcp_server._tool_manager
        tool_names = set(tool_manager._tools.keys())
        assert "run_task" in tool_names
        assert "get_task" in tool_names
        assert "build_uv_script_image_remote" in tool_names
        assert "flyte_uv_script_format" in tool_names
        assert "get_run" not in tool_names

    def test_total_tool_count(self):
        assert len(ALL_MCP_TOOLS) == 21
