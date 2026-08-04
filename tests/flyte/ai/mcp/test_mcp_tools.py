"""
Unit tests for the expanded Flyte MCP tool surface.

Covers:
- Tool annotations: every registered tool carries a title and a complete set of hints.
- Registration/filtering: groups, ``read_only`` derivation, and the explicit registration table
  (disabled tools are never added, rather than added-then-popped).
- ``get_logs`` bounding: line cap, truncation marker, and the immediate status string when logs
  are not available yet.
- ``search_*`` regex handling and its literal-substring fallback.
- ``get_run_io`` surfacing output-decode failures instead of swallowing them.
- Per-call project/domain resolution, including the central-mode ToolError.
- Tools calling the SDK the way the SDK is actually shaped (``Run.wait`` vs ``Run.watch``,
  ``Trigger.update`` vs a non-existent ``TriggerDetails.activate``).
- Allowlist enforcement on every path that can reach a task: ``run_task``, ``rerun_run``,
  ``list_tasks``.

Every remote call is monkeypatched; no test touches the network. Mocks are deliberately shaped
like the real SDK objects -- a mock that grows whatever attribute the tool reaches for would
not catch a tool reaching for an attribute that does not exist.
"""

from __future__ import annotations

import asyncio
import inspect
import pathlib
from datetime import timedelta
from types import SimpleNamespace
from typing import ClassVar

import pytest

import flyte
import flyte.remote
from flyte.ai.mcp import FlyteMCPAppEnvironment
from flyte.ai.mcp._flyte_mcp_app import (
    ALL_MCP_TOOLS,
    DOMAIN_ENV_VAR,
    PROJECT_ENV_VAR,
    READ_ONLY_MCP_TOOLS,
    TOOL_GROUP_MAPPING,
    TOOL_REGISTRY,
    ToolError,
    _resolve_tools,
    _search_files,
    resolve_tools,
)
from flyte.ai.mcp._tools import ALLOWLIST_SCAN_LIMIT, MAX_LOG_LINES, _collect_log_lines

# Tools the spec pins to specific hints; asserted individually so a careless registry edit shows up.
DESTRUCTIVE_TOOLS = {"abort_run", "abort_action", "delete_secret", "deactivate_app", "deactivate_trigger"}
WRITE_NON_DESTRUCTIVE_TOOLS = {
    "run_task",
    "rerun_run",
    "create_secret",
    "activate_app",
    "activate_trigger",
    "signal_condition",
}


def _tools_of(env: FlyteMCPAppEnvironment) -> dict:
    return env._mcp_server._tool_manager._tools


def _fn(env: FlyteMCPAppEnvironment, name: str):
    """The raw handler behind a registered tool."""
    return _tools_of(env)[name].fn


@pytest.fixture
def init_config(monkeypatch):
    """Install a module-global ``_InitConfig`` so project/domain-scoped tools can resolve."""
    import flyte._initialize as init_mod

    cfg = init_mod._InitConfig(
        root_dir=pathlib.Path.cwd(),
        org="test-org",
        project="cfg-project",
        domain="cfg-domain",
        client=object(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(init_mod, "_init_config", cfg)
    return cfg


# ------------------------------
# A1: annotations
# ------------------------------


class TestToolAnnotations:
    """Every registered tool must advertise a title and a full set of behavior hints."""

    def test_registry_covers_exactly_the_tool_literal(self):
        assert set(TOOL_REGISTRY) == set(ALL_MCP_TOOLS)

    def test_every_registered_tool_has_complete_annotations(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        for name, tool in _tools_of(env).items():
            ann = tool.annotations
            assert ann is not None, f"{name} has no annotations"
            assert ann.title, f"{name} has no title"
            assert ann.readOnlyHint is not None, f"{name} has no readOnlyHint"
            assert ann.destructiveHint is not None, f"{name} has no destructiveHint"
            assert ann.idempotentHint is not None, f"{name} has no idempotentHint"
            assert ann.openWorldHint is not None, f"{name} has no openWorldHint"

    def test_open_world_hint_is_false_everywhere(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert all(t.annotations.openWorldHint is False for t in _tools_of(env).values())

    def test_read_only_tools_are_annotated_read_only(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        for name in READ_ONLY_MCP_TOOLS:
            assert _tools_of(env)[name].annotations.readOnlyHint is True

    def test_read_prefixed_tools_are_all_read_only(self):
        # get_*/list_*/search_*/whoami/wait_for_run are reads by construction.
        for name in ALL_MCP_TOOLS:
            if name.startswith(("get_", "list_", "search_")) or name in ("whoami", "wait_for_run"):
                assert TOOL_REGISTRY[name].read_only is True, name

    def test_destructive_tools(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        for name, tool in _tools_of(env).items():
            assert tool.annotations.destructiveHint is (name in DESTRUCTIVE_TOOLS), name

    def test_write_non_destructive_tools(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        for name in WRITE_NON_DESTRUCTIVE_TOOLS:
            ann = _tools_of(env)[name].annotations
            assert ann.readOnlyHint is False
            assert ann.destructiveHint is False

    def test_lifecycle_tools_are_idempotent(self):
        for name in ALL_MCP_TOOLS:
            if name.startswith(("activate_", "deactivate_", "delete_", "abort_")):
                assert TOOL_REGISTRY[name].idempotent is True, name

    def test_annotations_survive_a_filtered_server(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["secret"])
        assert _tools_of(env)["delete_secret"].annotations.destructiveHint is True


# ------------------------------
# A5: groups & read_only
# ------------------------------


class TestGroups:
    """The new groups resolve to exactly the tools declared for them."""

    @pytest.mark.parametrize(
        ("group", "expected"),
        [
            ("action", {"list_actions", "get_action", "abort_action"}),
            ("logs", {"get_logs"}),
            ("project", {"list_projects", "get_project"}),
            ("secret", {"list_secrets", "create_secret", "delete_secret"}),
            ("condition", {"list_conditions", "signal_condition"}),
            ("identity", {"whoami"}),
            (
                "run",
                {"get_run", "get_run_io", "abort_run", "list_runs", "wait_for_run", "rerun_run"},
            ),
            ("app", {"get_app", "list_apps", "activate_app", "deactivate_app"}),
            (
                "trigger",
                {"list_triggers", "get_trigger", "activate_trigger", "deactivate_trigger"},
            ),
        ],
    )
    def test_group_membership(self, group, expected):
        assert resolve_tools([group], None) == expected
        assert set(TOOL_GROUP_MAPPING[group]) == expected

    def test_all_group_is_everything(self):
        assert resolve_tools(["all"], None) == set(ALL_MCP_TOOLS)

    def test_new_groups_register_their_tools(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["action", "logs", "identity"])
        assert set(_tools_of(env)) == {"list_actions", "get_action", "abort_action", "get_logs", "whoami"}


class TestReadOnlyDerivation:
    """``read_only=True`` keeps exactly the tools annotated ``readOnlyHint=True``."""

    def test_read_only_defaults_off(self):
        assert FlyteMCPAppEnvironment(name="test-mcp").read_only is False

    def test_read_only_keeps_only_read_tools(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", read_only=True)
        assert env.enabled_tools == set(READ_ONLY_MCP_TOOLS)
        assert set(_tools_of(env)) == set(READ_ONLY_MCP_TOOLS)

    def test_read_only_drops_every_mutating_tool(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", read_only=True)
        for name in DESTRUCTIVE_TOOLS | WRITE_NON_DESTRUCTIVE_TOOLS:
            assert name not in _tools_of(env)

    def test_read_only_intersects_with_groups(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["secret"], read_only=True)
        assert env.enabled_tools == {"list_secrets"}

    def test_read_only_intersects_with_explicit_tools(self):
        assert resolve_tools(None, ["get_run", "abort_run"], read_only=True) == {"get_run"}

    def test_read_only_derivation_matches_annotations(self):
        assert set(READ_ONLY_MCP_TOOLS) == {n for n, i in TOOL_REGISTRY.items() if i.read_only}


# ------------------------------
# A3: registration table
# ------------------------------


class TestRegistrationTable:
    """Disabled tools are never registered, instead of being popped back out afterwards."""

    def test_registered_names_match_enabled_names(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["logs", "project"])
        assert set(_tools_of(env)) == env.enabled_tools

    def test_core_group_registers_nothing(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["core"])
        assert _tools_of(env) == {}

    def test_deprecated_alias_still_resolves(self):
        assert _resolve_tools(["identity"], None) == {"whoami"}
        assert _resolve_tools(None, None) == set(ALL_MCP_TOOLS)

    def test_public_resolve_tools_is_exported(self):
        import flyte.ai.mcp as mcp_pkg

        assert mcp_pkg.resolve_tools(["logs"], None) == {"get_logs"}


# ------------------------------
# A3: search
# ------------------------------


class TestSearchRegex:
    """``search_*`` patterns are regexes, with a literal fallback and a match-count header."""

    @pytest.mark.asyncio
    async def test_regex_pattern_matches(self, tmp_path):
        (tmp_path / "a.py").write_text("import flyte\nimport flytekit\n")
        result = await _search_files(r"^import flyte$", str(tmp_path))
        assert "a.py" in result
        assert "1 files matched" in result

    @pytest.mark.asyncio
    async def test_regex_alternation(self, tmp_path):
        (tmp_path / "a.py").write_text("alpha\n")
        (tmp_path / "b.py").write_text("beta\n")
        result = await _search_files(r"alpha|beta", str(tmp_path))
        assert "2 files matched; showing top 2" in result

    @pytest.mark.asyncio
    async def test_invalid_regex_falls_back_to_literal(self, tmp_path):
        (tmp_path / "a.py").write_text("value = arr[0\n")
        result = await _search_files("arr[0", str(tmp_path))
        assert "a.py" in result
        assert "literal substring" in result

    @pytest.mark.asyncio
    async def test_header_reports_total_and_shown(self, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("needle\n")
        result = await _search_files("needle", str(tmp_path), top_n=3)
        assert result.splitlines()[0] == "5 files matched; showing top 3"

    @pytest.mark.asyncio
    async def test_no_matches_is_unchanged(self, tmp_path):
        (tmp_path / "a.py").write_text("hello\n")
        assert await _search_files("nothing-here", str(tmp_path)) == "No matches found"


# ------------------------------
# A2/A6: get_logs bounding
# ------------------------------


async def _agen(lines):
    for line in lines:
        yield line


class _FakeDetails:
    def __init__(self, *, logs_available: bool = True, phase: str = "RUNNING"):
        self._logs_available = logs_available
        self.phase = phase

    def logs_available(self, attempt=None) -> bool:
        return self._logs_available


class _FakeGetLogs:
    def __init__(self, lines, on_call=None):
        self._lines = lines
        self._on_call = on_call

    def aio(self, **kwargs):
        if self._on_call is not None:
            self._on_call(kwargs)
        return _agen(self._lines)


class _FakeAction:
    def __init__(self, details, lines, on_call=None):
        self._details = details
        self.get_logs = _FakeGetLogs(lines, on_call)

    async def details(self):
        return self._details


class _FakeActionGet:
    def __init__(self, action):
        self._action = action

    async def aio(self, **kwargs):
        return self._action


@pytest.fixture
def logs_env(init_config):
    return FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["logs"])


class TestCollectLogLines:
    """The bounding helper on its own: cap, marker, and the empty case."""

    @pytest.mark.asyncio
    async def test_truncation_marker_when_capped(self):
        out = await _collect_log_lines(_agen([f"line-{i}" for i in range(50)]), 10, 30.0)
        assert out.splitlines()[:10] == [f"line-{i}" for i in range(10)]
        assert out.endswith("...truncated at 10 lines")

    @pytest.mark.asyncio
    async def test_no_marker_when_stream_ends_first(self):
        out = await _collect_log_lines(_agen(["a", "b"]), 10, 30.0)
        assert out == "a\nb"

    @pytest.mark.asyncio
    async def test_empty_stream_reports_no_lines(self):
        out = await _collect_log_lines(_agen([]), 10, 30.0)
        assert out == "No log lines were returned."

    @pytest.mark.asyncio
    async def test_partial_output_survives_the_timeout(self):
        import asyncio

        async def _slow():
            yield "first"
            await asyncio.sleep(10)
            yield "never"

        out = await _collect_log_lines(_slow(), 100, 0.05)
        assert out.startswith("first")
        assert "stopped after" in out


class TestGetLogsTool:
    @pytest.mark.asyncio
    async def test_truncates_at_max_lines(self, logs_env, monkeypatch):
        action = _FakeAction(_FakeDetails(), [f"line-{i}" for i in range(500)])
        monkeypatch.setattr(flyte.remote.Action, "get", _FakeActionGet(action))

        out = await _fn(logs_env, "get_logs")(run_name="r1", action_name="a1", max_lines=5)
        assert out.splitlines()[:5] == [f"line-{i}" for i in range(5)]
        assert "...truncated at 5 lines" in out

    @pytest.mark.asyncio
    async def test_max_lines_is_hard_capped(self, logs_env, monkeypatch):
        action = _FakeAction(_FakeDetails(), [f"line-{i}" for i in range(MAX_LOG_LINES + 50)])
        monkeypatch.setattr(flyte.remote.Action, "get", _FakeActionGet(action))

        out = await _fn(logs_env, "get_logs")(run_name="r1", action_name="a1", max_lines=1_000_000)
        assert f"...truncated at {MAX_LOG_LINES} lines" in out
        assert len(out.splitlines()) == MAX_LOG_LINES + 1

    @pytest.mark.asyncio
    async def test_status_string_when_logs_not_available(self, logs_env, monkeypatch):
        consumed = []
        action = _FakeAction(
            _FakeDetails(logs_available=False, phase="QUEUED"),
            ["never-read"],
            on_call=consumed.append,
        )
        monkeypatch.setattr(flyte.remote.Action, "get", _FakeActionGet(action))

        out = await _fn(logs_env, "get_logs")(run_name="r1", action_name="a1")
        assert "not available yet" in out
        assert "QUEUED" in out
        assert "wait_for_logs" in out
        # The stream must not even be opened -- that is what keeps a queued action from blocking.
        assert consumed == []

    @pytest.mark.asyncio
    async def test_wait_for_logs_opts_into_blocking(self, logs_env, monkeypatch):
        action = _FakeAction(_FakeDetails(logs_available=False), ["a", "b"])
        monkeypatch.setattr(flyte.remote.Action, "get", _FakeActionGet(action))

        out = await _fn(logs_env, "get_logs")(run_name="r1", action_name="a1", wait_for_logs=True)
        assert out == "a\nb"

    @pytest.mark.asyncio
    async def test_filter_system_is_forwarded(self, logs_env, monkeypatch):
        seen = []
        action = _FakeAction(_FakeDetails(), ["x"], on_call=seen.append)
        monkeypatch.setattr(flyte.remote.Action, "get", _FakeActionGet(action))

        await _fn(logs_env, "get_logs")(run_name="r1", action_name="a1", attempt=2, filter_system=False)
        assert seen == [{"attempt": 2, "filter_system": False}]

    @pytest.mark.asyncio
    async def test_root_action_used_when_action_name_omitted(self, logs_env, monkeypatch):
        action = _FakeAction(_FakeDetails(), ["root-line"])

        class _FakeRun:
            def __init__(self):
                self.action = action
                self.get_logs = action.get_logs

        class _FakeRunGet:
            async def aio(self, **kwargs):
                assert kwargs == {"name": "r1"}
                return _FakeRun()

        monkeypatch.setattr(flyte.remote.Run, "get", _FakeRunGet())
        out = await _fn(logs_env, "get_logs")(run_name="r1")
        assert out == "root-line"


# ------------------------------
# A3: get_run_io error surfacing
# ------------------------------


class _FakeOutputs:
    named_outputs: ClassVar[dict] = {"o0": 1}


class _FakeRunIO:
    def __init__(self, *, outputs_exc: Exception | None = None, done: bool = True):
        self.name = "r1"
        self._outputs_exc = outputs_exc
        self._done = done
        self.inputs = _Aio(lambda: {"x": 1})
        self.outputs = _Aio(self._outputs)

    def _outputs(self):
        if self._outputs_exc is not None:
            raise self._outputs_exc
        return _FakeOutputs()

    def done(self) -> bool:
        return self._done


class _Aio:
    def __init__(self, fn):
        self._fn = fn

    async def aio(self, *args, **kwargs):
        return self._fn()


class TestGetRunIO:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["run"])

    @pytest.mark.asyncio
    async def test_output_error_is_surfaced(self, env, monkeypatch):
        run = _FakeRunIO(outputs_exc=RuntimeError("cannot guess python type"))
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(lambda: run))

        result = await _fn(env, "get_run_io")(name="r1")
        assert result["outputs"] is None
        assert result["outputs_error"] == "RuntimeError: cannot guess python type"

    @pytest.mark.asyncio
    async def test_no_error_key_on_success(self, env, monkeypatch):
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(_FakeRunIO))

        result = await _fn(env, "get_run_io")(name="r1")
        assert result["outputs"] == {"o0": 1}
        assert "outputs_error" not in result

    @pytest.mark.asyncio
    async def test_running_run_has_no_outputs_and_no_error(self, env, monkeypatch):
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(lambda: _FakeRunIO(done=False)))

        result = await _fn(env, "get_run_io")(name="r1")
        assert result["outputs"] is None
        assert "outputs_error" not in result


# ------------------------------
# A4: per-call project/domain
# ------------------------------


class TestScopeResolution:
    def test_explicit_arguments_win(self, init_config, monkeypatch):
        monkeypatch.setenv(PROJECT_ENV_VAR, "env-project")
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.resolve_scope("arg-project", "arg-domain") == ("arg-project", "arg-domain")

    def test_env_defaults_used_when_not_central(self, init_config, monkeypatch):
        monkeypatch.setenv(PROJECT_ENV_VAR, "env-project")
        monkeypatch.setenv(DOMAIN_ENV_VAR, "env-domain")
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.resolve_scope(None, None) == ("env-project", "env-domain")

    def test_init_config_is_the_last_fallback(self, init_config, monkeypatch):
        monkeypatch.delenv(PROJECT_ENV_VAR, raising=False)
        monkeypatch.delenv(DOMAIN_ENV_VAR, raising=False)
        env = FlyteMCPAppEnvironment(name="test-mcp")
        assert env.resolve_scope(None, None) == ("cfg-project", "cfg-domain")

    def test_central_mode_ignores_the_env_defaults(self, monkeypatch):
        import flyte._initialize as init_mod

        monkeypatch.setenv(PROJECT_ENV_VAR, "env-project")
        monkeypatch.setenv(DOMAIN_ENV_VAR, "env-domain")
        monkeypatch.setattr(init_mod, "_init_config", None)

        env = FlyteMCPAppEnvironment(name="test-mcp", central_mode=True)
        with pytest.raises(ToolError, match="multi-tenant"):
            env.resolve_scope(None, None)

    def test_central_mode_accepts_explicit_arguments(self, monkeypatch):
        import flyte._initialize as init_mod

        monkeypatch.setattr(init_mod, "_init_config", None)
        env = FlyteMCPAppEnvironment(name="test-mcp", central_mode=True)
        assert env.resolve_scope("p", "d") == ("p", "d")

    def test_missing_scope_names_the_env_vars_when_not_central(self, monkeypatch):
        import flyte._initialize as init_mod

        monkeypatch.delenv(PROJECT_ENV_VAR, raising=False)
        monkeypatch.delenv(DOMAIN_ENV_VAR, raising=False)
        monkeypatch.setattr(init_mod, "_init_config", None)

        env = FlyteMCPAppEnvironment(name="test-mcp")
        with pytest.raises(ToolError, match=PROJECT_ENV_VAR):
            env.resolve_scope(None, None)

    def test_scoped_installs_the_override_on_the_init_config(self, init_config):
        from flyte._initialize import _get_init_config

        env = FlyteMCPAppEnvironment(name="test-mcp")
        with env._scoped("other-project", "other-domain") as (project, domain):
            assert (project, domain) == ("other-project", "other-domain")
            cfg = _get_init_config()
            assert (cfg.project, cfg.domain) == ("other-project", "other-domain")
            # The override must carry the tenant identity through, not just the scope.
            assert cfg.org == "test-org"
        assert _get_init_config().project == "cfg-project"

    def test_scoped_is_a_no_op_when_the_scope_already_matches(self, init_config):
        from flyte._initialize import _get_init_config

        env = FlyteMCPAppEnvironment(name="test-mcp")
        with env._scoped("cfg-project", "cfg-domain"):
            assert _get_init_config() is init_config

    def test_scoped_without_any_config_is_a_tool_error(self, monkeypatch):
        import flyte._initialize as init_mod

        monkeypatch.setattr(init_mod, "_init_config", None)
        env = FlyteMCPAppEnvironment(name="test-mcp")
        with pytest.raises(ToolError, match="not initialized"):
            with env._scoped("p", "d"):
                pass

    @pytest.mark.asyncio
    async def test_scoped_isolates_concurrent_tool_calls(self, init_config):
        import asyncio

        from flyte._initialize import _get_init_config

        env = FlyteMCPAppEnvironment(name="test-mcp")
        seen: dict[str, str] = {}

        async def call(project: str) -> None:
            with env._scoped(project, "d"):
                await asyncio.sleep(0.01)
                seen[project] = _get_init_config().project

        await asyncio.gather(call("p1"), call("p2"))
        assert seen == {"p1": "p1", "p2": "p2"}

    @pytest.mark.asyncio
    async def test_tool_scopes_before_calling_the_sdk(self, init_config, monkeypatch):
        from flyte._initialize import _get_init_config

        observed = {}

        class _FakeRunGet:
            async def aio(self, **kwargs):
                cfg = _get_init_config()
                observed["project"] = cfg.project
                observed["domain"] = cfg.domain

                class _R:
                    name = "r1"
                    phase = "SUCCEEDED"
                    url = "https://example.com/r1"

                    def done(self):
                        return True

                return _R()

        monkeypatch.setattr(flyte.remote.Run, "get", _FakeRunGet())
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["run"])
        result = await _fn(env, "get_run")(name="r1", project="tool-project", domain="tool-domain")

        assert observed == {"project": "tool-project", "domain": "tool-domain"}
        assert result["phase"] == "SUCCEEDED"


class TestScopedToolSignatures:
    """Project/domain-scoped tools must actually accept the arguments."""

    SCOPED: ClassVar[list[str]] = [
        "get_run",
        "get_run_io",
        "abort_run",
        "list_runs",
        "wait_for_run",
        "rerun_run",
        "list_actions",
        "get_action",
        "abort_action",
        "get_logs",
        "get_app",
        "list_apps",
        "activate_app",
        "deactivate_app",
        "list_triggers",
        "get_trigger",
        "activate_trigger",
        "deactivate_trigger",
        "list_secrets",
        "create_secret",
        "delete_secret",
        "list_conditions",
        "signal_condition",
        "list_tasks",
    ]

    @pytest.mark.parametrize("name", SCOPED)
    def test_tool_takes_project_and_domain(self, name):
        import inspect

        env = FlyteMCPAppEnvironment(name="test-mcp")
        params = inspect.signature(_fn(env, name)).parameters
        assert "project" in params, name
        assert "domain" in params, name

    @pytest.mark.parametrize("name", ["list_projects", "get_project", "whoami"])
    def test_unscoped_tools_take_neither(self, name):
        import inspect

        env = FlyteMCPAppEnvironment(name="test-mcp")
        params = inspect.signature(_fn(env, name)).parameters
        assert "project" not in params
        assert "domain" not in params


class TestListOrdering:
    """Time-ordered list tools must request newest-first from the server.

    The control plane's default sort is ("created_at", "asc"), and `limit` truncates
    server-side — so without an explicit descending sort, the newest entries are the
    ones silently dropped and "recent" answers are stale.
    """

    CASES: ClassVar[list[tuple[str, str, dict]]] = [
        ("list_runs", "Run", {}),
        ("list_tasks", "Task", {}),
        ("list_apps", "App", {}),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("tool_name", "cls_name", "extra"), CASES)
    async def test_requests_newest_first(self, init_config, monkeypatch, tool_name, cls_name, extra):
        captured: dict = {}

        class _Listall:
            def aio(self, **kwargs):
                captured.update(kwargs)

                async def _gen():
                    return
                    yield

                return _gen()

        monkeypatch.setattr(getattr(flyte.remote, cls_name), "listall", _Listall())
        env = FlyteMCPAppEnvironment(name="test-mcp")
        await _fn(env, tool_name)(project="p", domain="d", **extra)

        assert captured.get("sort_by") == ("created_at", "desc"), tool_name


# ------------------------------
# wait_for_run: Run.wait, not Run.watch
# ------------------------------


class _FakeWait:
    """``Run.wait`` as the SDK exposes it: syncified (so ``.aio(...)``) and returning None.

    Waiting mutates the run in place rather than handing back a new one, which is why the tool
    has to read the phase off ``run`` afterwards.
    """

    def __init__(self, run, *, delay: float = 0.0, final_phase: str = "SUCCEEDED"):
        self._run = run
        self._delay = delay
        self._final_phase = final_phase
        self.calls: list[dict] = []

    async def aio(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self._delay:
            await asyncio.sleep(self._delay)
        self._run.phase = self._final_phase


class _FakeWaitableRun:
    """Shaped like ``flyte.remote.Run``: ``watch`` is a bare async generator with no ``.aio``.

    That is the whole point of this fake -- the old implementation called ``run.watch.aio(...)``,
    which cannot work against the real object and blows up here too.
    """

    def __init__(self, phase: str = "RUNNING", *, delay: float = 0.0, final_phase: str = "SUCCEEDED"):
        self.name = "r1"
        self.url = "https://example.com/r1"
        self.phase = phase
        self.wait = _FakeWait(self, delay=delay, final_phase=final_phase)

    async def watch(self, cache_data_on_done: bool = False):
        yield self

    def done(self) -> bool:
        return self.phase in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT")


class _SeqGet:
    """``X.get`` returning a scripted sequence, so re-fetches are observable."""

    def __init__(self, *values):
        self._values = list(values)
        self.calls: list[dict] = []

    async def aio(self, **kwargs):
        self.calls.append(kwargs)
        return self._values[min(len(self.calls) - 1, len(self._values) - 1)]


class TestWaitForRun:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["run"])

    def test_run_watch_is_a_bare_async_generator_on_the_real_sdk(self):
        # Pins the API the tool has to code against: `watch` is not syncified and takes no
        # interval/timeout, so `run.watch.aio(interval=..., timeout=...)` can never work.
        assert inspect.isasyncgenfunction(flyte.remote.Run.watch)
        assert not hasattr(flyte.remote.Run.watch, "aio")
        assert hasattr(flyte.remote.Run.wait, "aio")

    def test_poll_interval_is_not_advertised(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        params = inspect.signature(_fn(env, "wait_for_run")).parameters
        # There is no poll knob on `Run.wait`, so the tool must not pretend to have one.
        assert "poll_interval_s" not in params
        assert "timeout_s" in params

    @pytest.mark.asyncio
    async def test_terminal_run_is_reported_after_waiting(self, env, monkeypatch):
        run = _FakeWaitableRun()
        getter = _SeqGet(run)
        monkeypatch.setattr(flyte.remote.Run, "get", getter)

        result = await _fn(env, "wait_for_run")(name="r1")
        assert result == {
            "name": "r1",
            "phase": "SUCCEEDED",
            "url": "https://example.com/r1",
            "done": True,
        }
        assert run.wait.calls == [{"quiet": True}]
        assert len(getter.calls) == 1

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_status_instead_of_raising(self, env, monkeypatch):
        slow = _FakeWaitableRun(delay=10)
        still_running = _FakeWaitableRun(phase="RUNNING")
        getter = _SeqGet(slow, still_running)
        monkeypatch.setattr(flyte.remote.Run, "get", getter)

        result = await _fn(env, "wait_for_run")(name="r1", timeout_s=0.05)
        assert result["timed_out"] is True
        assert result["phase"] == "RUNNING"
        assert result["done"] is False
        assert "abort_run" in result["message"]
        # The run is re-fetched after the cancelled wait so the phase is not the stale one.
        assert len(getter.calls) == 2


# ------------------------------
# activate/deactivate_trigger: Trigger.update, not TriggerDetails.activate
# ------------------------------


class _TriggerDetailsLike:
    """A stand-in for what ``Trigger.get`` actually returns: no activate/deactivate on it."""

    __slots__ = ("automation_spec", "is_active", "name", "task_name")

    def __init__(self):
        self.name = "nightly"
        self.task_name = "t"
        self.is_active = False
        self.automation_spec = None


class _UpdateRecorder:
    def __init__(self):
        self.calls: list[dict] = []

    async def aio(self, **kwargs):
        self.calls.append(kwargs)


class TestTriggerActivation:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["trigger"])

    def test_trigger_get_returns_details_with_no_activation_methods(self):
        from flyte.remote._trigger import TriggerDetails

        # `Trigger.get` -> `TriggerDetails`, which has no activate/deactivate at all; the only
        # way to flip a trigger is the `Trigger.update` classmethod.
        assert not hasattr(TriggerDetails, "activate")
        assert not hasattr(TriggerDetails, "deactivate")
        assert hasattr(flyte.remote.Trigger, "update")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "active", "result_key"),
        [("activate_trigger", True, "activated"), ("deactivate_trigger", False, "deactivated")],
    )
    async def test_flips_active_via_update(self, env, monkeypatch, tool_name, active, result_key):
        recorder = _UpdateRecorder()
        monkeypatch.setattr(flyte.remote.Trigger, "update", recorder)
        # Present but useless, exactly like the real thing: reaching for `.activate` on it fails.
        monkeypatch.setattr(flyte.remote.Trigger, "get", _SeqGet(_TriggerDetailsLike()))

        result = await _fn(env, tool_name)(task_name="t", trigger_name="nightly", project="p", domain="d")

        assert recorder.calls == [{"name": "nightly", "task_name": "t", "active": active}]
        assert result == {"task_name": "t", "name": "nightly", result_key: True}

    @pytest.mark.asyncio
    async def test_allowlist_blocks_before_any_call(self, init_config, monkeypatch):
        recorder = _UpdateRecorder()
        monkeypatch.setattr(flyte.remote.Trigger, "update", recorder)
        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["trigger"], trigger_allowlist=["t/other"])

        with pytest.raises(ToolError, match="not allowlisted"):
            await _fn(env, "activate_trigger")(task_name="t", trigger_name="nightly", project="p", domain="d")
        assert recorder.calls == []


# ------------------------------
# rerun_run allowlist gate
# ------------------------------


class _RerunRecorder:
    def __init__(self):
        self.calls: list[tuple] = []

    async def aio(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(name="r2", url="https://example.com/r2")


def _run_with_task(task_name: str | None):
    """A ``Run`` stand-in exposing the task behind it the way the SDK does: ``run.action.task_name``."""
    return SimpleNamespace(name="r1", action=SimpleNamespace(task_name=task_name))


class TestRerunAllowlist:
    """`rerun_run` launches a task, so it must clear the same gate as `run_task`."""

    def test_run_exposes_its_task_name_through_action(self):
        from flyte.remote._action import Action

        # The attribute the gate reads lives on the run's action, not on the run itself.
        assert isinstance(Action.task_name, property)

    @pytest.mark.asyncio
    async def test_disallowed_task_is_rejected_before_rerunning(self, init_config, monkeypatch):
        rerun = _RerunRecorder()
        monkeypatch.setattr(flyte, "rerun", rerun)
        getter = _SeqGet(_run_with_task("forbidden-task"))
        monkeypatch.setattr(flyte.remote.Run, "get", getter)

        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["run"], task_allowlist=["d/p/allowed-task"])
        with pytest.raises(ToolError, match="not allowlisted"):
            await _fn(env, "rerun_run")(run_name="r1", project="p", domain="d")
        assert rerun.calls == []

    @pytest.mark.asyncio
    async def test_allowed_task_reruns(self, init_config, monkeypatch):
        rerun = _RerunRecorder()
        monkeypatch.setattr(flyte, "rerun", rerun)
        monkeypatch.setattr(flyte.remote.Run, "get", _SeqGet(_run_with_task("allowed-task")))

        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["run"], task_allowlist=["d/p/allowed-task"])
        result = await _fn(env, "rerun_run")(run_name="r1", project="p", domain="d")

        assert result == {"name": "r2", "url": "https://example.com/r2"}
        assert rerun.calls == [(("r1",), {})]

    @pytest.mark.asyncio
    async def test_unknown_task_name_is_rejected_under_an_allowlist(self, init_config, monkeypatch):
        rerun = _RerunRecorder()
        monkeypatch.setattr(flyte, "rerun", rerun)
        monkeypatch.setattr(flyte.remote.Run, "get", _SeqGet(_run_with_task(None)))

        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["run"], task_allowlist=["d/p/allowed-task"])
        with pytest.raises(ToolError, match="Cannot determine which task"):
            await _fn(env, "rerun_run")(run_name="r1", project="p", domain="d")
        assert rerun.calls == []

    @pytest.mark.asyncio
    async def test_no_allowlist_skips_the_lookup(self, init_config, monkeypatch):
        rerun = _RerunRecorder()
        monkeypatch.setattr(flyte, "rerun", rerun)
        getter = _SeqGet(_run_with_task("anything"))
        monkeypatch.setattr(flyte.remote.Run, "get", getter)

        env = FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["run"])
        await _fn(env, "rerun_run")(run_name="r1", project="p", domain="d")

        assert rerun.calls == [(("r1",), {})]
        assert getter.calls == []  # no allowlist, no reason to pay for the extra round trip


# ------------------------------
# create_secret type validation
# ------------------------------


class TestCreateSecretType:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["secret"])

    @pytest.mark.asyncio
    async def test_regular_is_created(self, env, monkeypatch):
        recorder = _UpdateRecorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        result = await _fn(env, "create_secret")(name="s", value="v", project="p", domain="d")
        assert result == {"name": "s", "created": True}
        assert recorder.calls == [{"name": "s", "value": "v", "type": "regular"}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_type", ["image_pull", "IMAGE_PULL", "reglar", ""])
    async def test_non_regular_types_are_rejected(self, env, monkeypatch, bad_type):
        recorder = _UpdateRecorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        # `Secret.create` treats every non-"regular" value as image_pull, so a typo would
        # otherwise silently take a path this server can never satisfy.
        with pytest.raises(ToolError, match="Unsupported secret type"):
            await _fn(env, "create_secret")(name="s", value="v", type=bad_type, project="p", domain="d")
        assert recorder.calls == []


# ------------------------------
# allowlist-aware listing
# ------------------------------


class _CapturingListall:
    def __init__(self, items):
        self._items = items
        self.captured: dict = {}

    def aio(self, **kwargs):
        self.captured.update(kwargs)

        async def _gen():
            for item in self._items:
                yield item

        return _gen()


def _tasks(*names):
    return [SimpleNamespace(name=n, version="v1") for n in names]


class TestListTasksAllowlist:
    @pytest.mark.asyncio
    async def test_only_allowlisted_tasks_are_listed(self, init_config, monkeypatch):
        listall = _CapturingListall(_tasks("a", "b", "c"))
        monkeypatch.setattr(flyte.remote.Task, "listall", listall)

        env = FlyteMCPAppEnvironment(name="test-mcp", task_allowlist=["d/p/a", "d/p/c"])
        out = await _fn(env, "list_tasks")(project="p", domain="d")

        assert [t["name"] for t in out] == ["a", "c"]

    @pytest.mark.asyncio
    async def test_everything_is_listed_without_an_allowlist(self, init_config, monkeypatch):
        listall = _CapturingListall(_tasks("a", "b", "c"))
        monkeypatch.setattr(flyte.remote.Task, "listall", listall)

        env = FlyteMCPAppEnvironment(name="test-mcp")
        out = await _fn(env, "list_tasks")(project="p", domain="d", limit=100)

        assert [t["name"] for t in out] == ["a", "b", "c"]
        assert listall.captured["limit"] == 100

    @pytest.mark.asyncio
    async def test_allowlist_widens_the_server_side_scan(self, init_config, monkeypatch):
        listall = _CapturingListall(_tasks("a"))
        monkeypatch.setattr(flyte.remote.Task, "listall", listall)

        env = FlyteMCPAppEnvironment(name="test-mcp", task_allowlist=["d/p/a"])
        await _fn(env, "list_tasks")(project="p", domain="d", limit=10)

        # Filtering happens client-side, so a server-side limit of 10 would drop allowed tasks
        # sitting behind 10 disallowed ones.
        assert listall.captured["limit"] == ALLOWLIST_SCAN_LIMIT

    @pytest.mark.asyncio
    async def test_limit_still_bounds_the_result(self, init_config, monkeypatch):
        listall = _CapturingListall(_tasks(*[f"t{i}" for i in range(20)]))
        monkeypatch.setattr(flyte.remote.Task, "listall", listall)

        env = FlyteMCPAppEnvironment(name="test-mcp", task_allowlist=[f"d/p/t{i}" for i in range(20)])
        out = await _fn(env, "list_tasks")(project="p", domain="d", limit=3)

        assert [t["name"] for t in out] == ["t0", "t1", "t2"]


class TestListAppsAllowlist:
    @staticmethod
    def _apps(*names):
        return [SimpleNamespace(name=n, deployment_status=0, endpoint=f"{n}.example.com", url=None) for n in names]

    @pytest.mark.asyncio
    async def test_allowlist_widens_the_server_side_scan(self, init_config, monkeypatch):
        listall = _CapturingListall(self._apps("a", "b"))
        monkeypatch.setattr(flyte.remote.App, "listall", listall)

        env = FlyteMCPAppEnvironment(name="test-mcp", app_allowlist=["b"])
        out = await _fn(env, "list_apps")(project="p", domain="d", limit=10)

        assert [a["name"] for a in out] == ["b"]
        assert listall.captured["limit"] == ALLOWLIST_SCAN_LIMIT

    @pytest.mark.asyncio
    async def test_no_allowlist_passes_the_callers_limit(self, init_config, monkeypatch):
        listall = _CapturingListall(self._apps("a"))
        monkeypatch.setattr(flyte.remote.App, "listall", listall)

        env = FlyteMCPAppEnvironment(name="test-mcp")
        await _fn(env, "list_apps")(project="p", domain="d", limit=10)

        assert listall.captured["limit"] == 10


# ------------------------------
# get_task interface
# ------------------------------


class _FakeTaskDetails:
    def __init__(self, interface):
        self.name = "t"
        self.version = "v1"
        self.task_type = "python"
        self.interface = interface
        self.required_args = ("x",)
        self.default_input_args = ("y",)
        self.cache = SimpleNamespace(behavior="disable", version_override=None, serialize=False)
        self.secrets = None

    @property
    def fetch(self):
        return _Aio(lambda: self)


class TestGetTaskInterface:
    @pytest.mark.asyncio
    async def test_interface_is_returned_so_inputs_can_be_constructed(self, init_config, monkeypatch):
        from flyte.models import NativeInterface

        iface = NativeInterface(
            inputs={"x": (int, inspect.Parameter.empty), "y": (str, "hello")},
            outputs={"o0": float},
        )
        td = _FakeTaskDetails(iface)
        monkeypatch.setattr(flyte.remote.Task, "get", lambda **kwargs: td)

        env = FlyteMCPAppEnvironment(name="test-mcp")
        result = await _fn(env, "get_task")(domain="d", project="p", name="t")

        assert result["interface"]["inputs"] == {"x": "int", "y": "str"}
        assert result["interface"]["outputs"] == {"o0": "float"}
        assert "x: int" in result["interface"]["signature"]
        assert result["required_args"] == ("x",)


# ------------------------------
# get_action per-attempt timing
# ------------------------------


class _FakeActionDetails:
    def __init__(self):
        from flyte.models import ActionPhase

        self.name = "a0"
        self.run_name = "r1"
        self.task_name = "t"
        self.phase = "FAILED"
        self.attempts = 2
        self.error_info = None
        self.abort_info = None
        self.runtime = timedelta(seconds=30)
        self.phase_durations = {ActionPhase.RUNNING: timedelta(seconds=20)}
        self.transitions_requested: list[int | None] = []
        self.logs_requested: list[int | None] = []

    def get_phase_transitions(self, attempt=None):
        from flyte.models import ActionPhase

        self.transitions_requested.append(attempt)
        return [SimpleNamespace(phase=ActionPhase.RUNNING, duration=timedelta(seconds=5))]

    def logs_available(self, attempt=None) -> bool:
        self.logs_requested.append(attempt)
        return True


class TestGetActionTiming:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp", tool_groups=["action"])

    def test_get_phase_transitions_takes_an_attempt(self):
        from flyte.remote._action import ActionDetails

        params = inspect.signature(ActionDetails.get_phase_transitions).parameters
        assert "attempt" in params

    @pytest.mark.asyncio
    async def test_latest_attempt_by_default(self, env, monkeypatch):
        details = _FakeActionDetails()
        monkeypatch.setattr(flyte.remote.ActionDetails, "get", _Aio(lambda: details))

        result = await _fn(env, "get_action")(run_name="r1", action_name="a0")
        assert result["timing"]["running_s"] == 20.0
        assert result["timing"]["total_s"] == 30.0
        assert result["timing"]["attempt"] == 2

    @pytest.mark.asyncio
    async def test_requested_attempt_drives_the_timing_too(self, env, monkeypatch):
        details = _FakeActionDetails()
        monkeypatch.setattr(flyte.remote.ActionDetails, "get", _Aio(lambda: details))

        result = await _fn(env, "get_action")(run_name="r1", action_name="a0", attempt=1)
        # Timing and logs_available must describe the same attempt, not attempt 1's logs next
        # to the latest attempt's durations.
        assert details.transitions_requested == [1]
        assert details.logs_requested == [1]
        assert result["timing"]["running_s"] == 5.0
        assert result["timing"]["attempt"] == 1


class TestSearchIsBounded:
    """Search patterns are caller-supplied, so a hostile one must not stall the server.

    `re` cannot bound backtracking and a catastrophic match is a single C call holding the
    GIL — moving it to a thread does not help. Matching goes through `regex`'s per-match
    timeout instead.
    """

    @pytest.fixture
    def corpus(self, tmp_path):
        (tmp_path / "f.txt").write_text("\n".join(["a" * 40 + "!"] * 60))
        return str(tmp_path)

    @pytest.mark.asyncio
    async def test_catastrophic_pattern_leaves_loop_responsive(self, corpus):
        import time as _time

        from flyte.ai.mcp._tools import _search_files

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        started = _time.monotonic()
        await _search_files("(a|a)+b", corpus)
        elapsed = _time.monotonic() - started
        t.cancel()

        # Unbounded, this pattern runs for minutes with the loop wedged the whole time.
        assert elapsed < 30, f"search took {elapsed:.1f}s"
        assert ticks > 10, "event loop was starved while the pattern backtracked"

    @pytest.mark.asyncio
    async def test_overlong_pattern_is_rejected(self, corpus):
        from flyte.ai.mcp._tools import MAX_SEARCH_PATTERN_LEN, _search_files

        out = await _search_files("a" * (MAX_SEARCH_PATTERN_LEN + 1), corpus)
        assert "too long" in out

    @pytest.mark.asyncio
    async def test_ordinary_search_still_matches(self, corpus):
        from flyte.ai.mcp._tools import _search_files

        assert "files matched" in await _search_files("a{5}", corpus)
