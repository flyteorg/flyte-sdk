"""
Unit tests for the "launch and operate" MCP tools: the run settings on ``run_task``, the new
``create_trigger`` / ``delete_trigger`` / ``delete_app`` tools, ``list_apps`` ``only_mine`` /
``status`` and ``list_tasks`` ``name``.

Each tool is checked three ways: the SDK is called with the right arguments (recorders), a bad or
disallowed call is a ``ToolError`` raised before any SDK call, and a call without the new
arguments reaches the SDK exactly as it did before they existed.

Every remote call is monkeypatched; nothing touches the network.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.app import app_definition_pb2

import flyte
import flyte.remote
from flyte.ai.mcp import FlyteMCPAppEnvironment
from flyte.ai.mcp._flyte_mcp_app import TOOL_REGISTRY, ToolError
from flyte.ai.mcp._tools import ALLOWLIST_SCAN_LIMIT


def _fn(env: FlyteMCPAppEnvironment, name: str):
    return env._mcp_server._tool_manager._tools[name].fn


def _env(group: str, **kwargs) -> FlyteMCPAppEnvironment:
    return FlyteMCPAppEnvironment(name="test-mcp", tool_groups=[group], **kwargs)


@pytest.fixture
def init_config(monkeypatch):
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


class _Aio:
    """A syncified SDK callable: ``X.get.aio(...)`` returning a canned value, recording its calls.

    ``scope`` records the project/domain the init config carried at call time, which is how
    the SDK entrypoints that take no project/domain arguments see the per-call scope.
    """

    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error
        self.calls: list[tuple[tuple, dict]] = []
        self.scope: list[tuple[str, str]] = []

    async def aio(self, *args, **kwargs):
        from flyte._initialize import get_init_config

        cfg = get_init_config()
        self.scope.append((cfg.project, cfg.domain))
        self.calls.append((args, kwargs))
        if self._error is not None:
            raise self._error
        return self._value


async def _agen(items):
    for item in items:
        yield item


class _ListAll:
    def __init__(self, items=()):
        self._items = list(items)
        self.calls: list[dict] = []

    def aio(self, **kwargs):
        self.calls.append(kwargs)
        return _agen(self._items)


# ------------------------------
# run_task
# ------------------------------


class _RunnerFactory:
    """``flyte.with_runcontext``: records its arguments and hands back a runner with a recording ``run``."""

    def __init__(self):
        self.calls: list[dict] = []
        self.run = _Aio(SimpleNamespace(name="r-ctx", url="https://example.com/r-ctx"))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(run=self.run)


@pytest.fixture
def run_sdk(init_config, monkeypatch):
    task_get = _Aio()
    tk = object()

    def get(**kwargs):
        task_get.calls.append(((), kwargs))
        return tk

    run = _Aio(SimpleNamespace(name="r1", url="https://example.com/r1"))
    runner = _RunnerFactory()
    monkeypatch.setattr(flyte.remote.Task, "get", get)
    monkeypatch.setattr(flyte, "run", run)
    monkeypatch.setattr(flyte, "with_runcontext", runner)
    return SimpleNamespace(task_get=task_get, tk=tk, run=run, runner=runner)


def _run_task(server=None, **kwargs):
    # `server` because `env` is a run_task argument (the run's environment variables).
    return _fn(server or _env("task"), "run_task")(domain="d", project="p", name="t", inputs={"x": 1}, **kwargs)


class TestRunTaskSettings:
    @pytest.mark.asyncio
    async def test_defaults_reproduce_the_previous_sdk_call(self, run_sdk):
        result = await _run_task()
        assert result == {"url": "https://example.com/r1", "name": "r1"}
        assert run_sdk.run.calls == [((run_sdk.tk,), {"x": 1})]
        assert run_sdk.runner.calls == []
        assert run_sdk.task_get.calls == [
            ((), {"project": "p", "domain": "d", "name": "t", "version": None, "auto_version": "latest"})
        ]

    @pytest.mark.asyncio
    async def test_empty_settings_are_the_same_as_none(self, run_sdk):
        await _run_task(run_name="", labels={}, env={}, queue="", custom_context={}, run_project="")
        assert run_sdk.runner.calls == []
        assert run_sdk.run.calls == [((run_sdk.tk,), {"x": 1})]

    @pytest.mark.asyncio
    async def test_every_setting_goes_through_with_runcontext(self, run_sdk):
        result = await _run_task(
            version="v1",
            run_name="nightly-1",
            labels={"team": "ml"},
            env={"LOG": "debug"},
            queue="gpu",
            service_account="runner",
            max_action_concurrency=4,
            raw_data_path="s3://bucket/raw",
            custom_context={"k": "v"},
            run_project="other-p",
            run_domain="other-d",
        )
        assert result == {"url": "https://example.com/r-ctx", "name": "r-ctx"}
        assert run_sdk.runner.calls == [
            {
                "mode": "remote",
                "name": "nightly-1",
                "env_vars": {"LOG": "debug"},
                "labels": {"team": "ml"},
                "queue": "gpu",
                "service_account": "runner",
                "raw_data_path": "s3://bucket/raw",
                "custom_context": {"k": "v"},
                "project": "other-p",
                "domain": "other-d",
                "max_action_concurrency": 4,
            }
        ]
        # The task is still looked up where it is registered, at the version asked for.
        assert run_sdk.task_get.calls[0][1]["version"] == "v1"
        assert run_sdk.task_get.calls[0][1]["project"] == "p"
        assert run_sdk.run.calls == []
        assert run_sdk.runner.run.calls == [((run_sdk.tk,), {"x": 1})]

    @pytest.mark.asyncio
    async def test_zero_concurrency_is_forwarded(self, run_sdk):
        # 0 means "platform default" to the SDK, which is still a value the caller chose.
        await _run_task(max_action_concurrency=0)
        assert run_sdk.runner.calls == [{"mode": "remote", "max_action_concurrency": 0}]

    @pytest.mark.asyncio
    async def test_only_the_settings_given_are_passed(self, run_sdk):
        await _run_task(queue="gpu")
        assert run_sdk.runner.calls == [{"mode": "remote", "queue": "gpu"}]

    @pytest.mark.asyncio
    async def test_env_parameter_does_not_hide_the_server_environment(self, run_sdk):
        # `env` is the run's environment variables; the allowlist must still come from the server.
        server = _env("task", task_allowlist=["d/p/t"])
        await _run_task(server, env={"A": "1"})
        assert run_sdk.runner.calls == [{"mode": "remote", "env_vars": {"A": "1"}}]


class TestRunTaskValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"labels": {"": "x"}}, "labels keys must not be empty"),
            ({"env": {"": "x"}}, "env keys must not be empty"),
            ({"custom_context": {"": "x"}}, "custom_context keys must not be empty"),
            ({"custom_context": {"_u_produced_artifacts": "x"}}, "reserved by Flyte"),
        ],
    )
    async def test_bad_key_value_settings_are_tool_errors(self, run_sdk, kwargs, message):
        with pytest.raises(ToolError, match=message):
            await _run_task(**kwargs)
        assert run_sdk.task_get.calls == []
        assert run_sdk.run.calls == []

    @pytest.mark.asyncio
    async def test_the_sdk_rejecting_the_run_context_is_a_tool_error(self, init_config, monkeypatch):
        # The real `with_runcontext`: concurrency 1 would deadlock a run, so the SDK refuses it.
        task_get = []
        monkeypatch.setattr(flyte.remote.Task, "get", lambda **kw: task_get.append(kw))
        with pytest.raises(ToolError, match="max_action_concurrency"):
            await _run_task(max_action_concurrency=1)
        assert task_get == []

    @pytest.mark.asyncio
    async def test_allowlist_denial_comes_before_any_sdk_call(self, run_sdk):
        with pytest.raises(ToolError, match="not allowlisted"):
            await _run_task(_env("task", task_allowlist=["other"]), queue="gpu")
        assert run_sdk.task_get.calls == []
        assert run_sdk.runner.calls == []
        assert run_sdk.run.calls == []


# ------------------------------
# create_trigger
# ------------------------------


@pytest.fixture
def trigger_create(init_config, monkeypatch):
    create = _Aio(SimpleNamespace(name="nightly", is_active=True))
    monkeypatch.setattr(flyte.remote.Trigger, "create", create)
    return create


def _create_trigger(env=None, **kwargs):
    kwargs.setdefault("schedule", "0 0 * * *")
    return _fn(env or _env("trigger"), "create_trigger")(task_name="my_env.my_task", trigger_name="nightly", **kwargs)


class TestCreateTrigger:
    @pytest.mark.asyncio
    async def test_builds_the_cli_trigger_with_defaults(self, trigger_create):
        result = await _create_trigger()
        assert result == {"task_name": "my_env.my_task", "name": "nightly", "created": True, "is_active": True}
        ((args, kwargs),) = trigger_create.calls
        (trigger,) = args
        assert trigger == flyte.Trigger(
            name="nightly", automation=flyte.Cron("0 0 * * *"), description="", auto_activate=True
        )
        assert kwargs == {"task_name": "my_env.my_task", "task_version": None}

    @pytest.mark.asyncio
    async def test_options_are_forwarded(self, trigger_create):
        await _create_trigger(
            description="runs nightly",
            auto_activate=False,
            trigger_time_input="when",
            task_version="v7",
        )
        ((args, kwargs),) = trigger_create.calls
        assert args[0] == flyte.Trigger(
            name="nightly",
            automation=flyte.Cron("0 0 * * *"),
            description="runs nightly",
            auto_activate=False,
            inputs={"when": flyte.TriggerTime},
        )
        assert kwargs == {"task_name": "my_env.my_task", "task_version": "v7"}

    @pytest.mark.asyncio
    async def test_runs_in_the_requested_project_and_domain(self, trigger_create):
        await _create_trigger(project="p2", domain="d2")
        assert trigger_create.scope == [("p2", "d2")]

    @pytest.mark.asyncio
    async def test_unusable_trigger_definition_is_a_tool_error(self, trigger_create):
        with pytest.raises(ToolError, match="name cannot be empty"):
            await _fn(_env("trigger"), "create_trigger")(task_name="t", trigger_name="", schedule="0 0 * * *")
        assert trigger_create.calls == []

    @pytest.mark.asyncio
    async def test_missing_task_is_a_tool_error(self, init_config, monkeypatch):
        # `Trigger.create` turns the control plane's NOT_FOUND into a ValueError.
        create = _Aio(error=ValueError("Task my_env.my_task:latest not found"))
        monkeypatch.setattr(flyte.remote.Trigger, "create", create)
        with pytest.raises(ToolError, match="not found"):
            await _create_trigger()

    @pytest.mark.asyncio
    async def test_a_cron_expression_the_server_rejects_is_a_tool_error(self, init_config, monkeypatch):
        # `flyte.Cron` does not validate; the control plane does.
        create = _Aio(error=ConnectError(Code.INVALID_ARGUMENT, "invalid cron expression"))
        monkeypatch.setattr(flyte.remote.Trigger, "create", create)
        with pytest.raises(ToolError, match="rejected the trigger: invalid cron expression"):
            await _create_trigger(schedule="not a cron")

    @pytest.mark.asyncio
    async def test_other_server_errors_are_not_disguised(self, init_config, monkeypatch):
        error = ConnectError(Code.UNAVAILABLE, "down")
        monkeypatch.setattr(flyte.remote.Trigger, "create", _Aio(error=error))
        with pytest.raises(ConnectError) as exc:
            await _create_trigger()
        assert exc.value is error

    @pytest.mark.asyncio
    async def test_trigger_allowlist_denial_comes_before_any_sdk_call(self, trigger_create):
        env = _env("trigger", trigger_allowlist=["my_env.my_task/other"])
        with pytest.raises(ToolError, match=r"Trigger my_env\.my_task/nightly is not allowlisted"):
            await _create_trigger(env)
        assert trigger_create.calls == []

    @pytest.mark.asyncio
    async def test_task_allowlist_denial_comes_before_any_sdk_call(self, trigger_create):
        # The trigger itself is allowed, but it would launch a task the server does not permit.
        env = _env("trigger", trigger_allowlist=["nightly"], task_allowlist=["cfg-domain/cfg-project/other"])
        with pytest.raises(ToolError, match=r"Task cfg-domain/cfg-project/my_env\.my_task is not allowlisted"):
            await _create_trigger(env)
        assert trigger_create.calls == []

    @pytest.mark.asyncio
    async def test_allowlisted_task_and_trigger_pass(self, trigger_create):
        env = _env("trigger", trigger_allowlist=["nightly"], task_allowlist=["my_env.my_task"])
        await _create_trigger(env)
        assert len(trigger_create.calls) == 1


# ------------------------------
# delete_trigger / delete_app
# ------------------------------


class TestDeleteTrigger:
    @pytest.fixture
    def delete(self, init_config, monkeypatch):
        delete = _Aio()
        monkeypatch.setattr(flyte.remote.Trigger, "delete", delete)
        return delete

    @pytest.mark.asyncio
    async def test_deletes_in_the_resolved_scope(self, delete):
        result = await _fn(_env("trigger"), "delete_trigger")(task_name="t", trigger_name="nightly")
        assert result == {"task_name": "t", "name": "nightly", "deleted": True}
        assert delete.calls == [
            ((), {"name": "nightly", "task_name": "t", "project": "cfg-project", "domain": "cfg-domain"})
        ]

    @pytest.mark.asyncio
    async def test_explicit_project_and_domain_win(self, delete):
        await _fn(_env("trigger"), "delete_trigger")(task_name="t", trigger_name="nightly", project="p2", domain="d2")
        assert delete.calls[0][1]["project"] == "p2"
        assert delete.calls[0][1]["domain"] == "d2"

    @pytest.mark.asyncio
    async def test_allowlist_denial_comes_before_any_sdk_call(self, delete):
        env = _env("trigger", trigger_allowlist=["t/other"])
        with pytest.raises(ToolError, match="Trigger t/nightly is not allowlisted"):
            await _fn(env, "delete_trigger")(task_name="t", trigger_name="nightly")
        assert delete.calls == []


class TestDeleteApp:
    @pytest.fixture
    def delete(self, init_config, monkeypatch):
        delete = _Aio()
        monkeypatch.setattr(flyte.remote.App, "delete", delete)
        return delete

    @pytest.mark.asyncio
    async def test_deletes_in_the_resolved_scope(self, delete):
        result = await _fn(_env("app"), "delete_app")(name="my-app")
        assert result == {"name": "my-app", "deleted": True}
        assert delete.calls == [((), {"name": "my-app", "project": "cfg-project", "domain": "cfg-domain"})]

    @pytest.mark.asyncio
    async def test_explicit_project_and_domain_win(self, delete):
        await _fn(_env("app"), "delete_app")(name="my-app", project="p2", domain="d2")
        assert delete.calls == [((), {"name": "my-app", "project": "p2", "domain": "d2"})]

    @pytest.mark.asyncio
    async def test_allowlist_denial_comes_before_any_sdk_call(self, delete):
        with pytest.raises(ToolError, match="App my-app is not allowlisted"):
            await _fn(_env("app", app_allowlist=["other"]), "delete_app")(name="my-app")
        assert delete.calls == []


# ------------------------------
# list_apps
# ------------------------------


def _app(name, status="DEPLOYMENT_STATUS_ACTIVE"):
    return SimpleNamespace(
        name=name,
        deployment_status=app_definition_pb2.Status.DeploymentStatus.Value(status),
        endpoint=f"https://{name}.example.com",
        url=f"https://console/{name}",
    )


class TestListAppsFilters:
    @pytest.fixture
    def listall(self, init_config, monkeypatch):
        listall = _ListAll([_app("a"), _app("b", "DEPLOYMENT_STATUS_FAILED")])
        monkeypatch.setattr(flyte.remote.App, "listall", listall)
        return listall

    @pytest.mark.asyncio
    async def test_defaults_reproduce_the_previous_sdk_call(self, listall, monkeypatch):
        user_get = _Aio(SimpleNamespace(subject=lambda: "sub-1"))
        monkeypatch.setattr(flyte.remote.User, "get", user_get)
        out = await _fn(_env("app"), "list_apps")()
        assert [a["name"] for a in out] == ["a", "b"]
        assert out[1]["status"] == "FAILED"
        assert listall.calls == [{"sort_by": ("created_at", "desc"), "limit": 100}]
        assert user_get.calls == []

    @pytest.mark.asyncio
    async def test_empty_status_is_unset(self, listall):
        await _fn(_env("app"), "list_apps")(status=[])
        assert listall.calls == [{"sort_by": ("created_at", "desc"), "limit": 100}]

    @pytest.mark.asyncio
    async def test_only_mine_filters_on_the_callers_subject(self, listall, monkeypatch):
        monkeypatch.setattr(flyte.remote.User, "get", _Aio(SimpleNamespace(subject=lambda: "sub-1")))
        await _fn(_env("app"), "list_apps")(only_mine=True)
        assert listall.calls[0]["created_by_subject"] == "sub-1"

    @pytest.mark.asyncio
    async def test_status_is_passed_through_for_the_sdk_to_normalise(self, listall):
        await _fn(_env("app"), "list_apps")(status=["Active", "FAILED"])
        assert listall.calls[0]["in_status"] == ("Active", "FAILED")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [["active", "exploded"], [""]])
    async def test_unknown_status_lists_the_valid_ones(self, listall, status):
        with pytest.raises(ToolError) as exc:
            await _fn(_env("app"), "list_apps")(status=status)
        assert "Valid statuses" in str(exc.value)
        assert "active" in str(exc.value)
        assert "unspecified" not in str(exc.value)
        assert listall.calls == []


# ------------------------------
# list_tasks
# ------------------------------


class TestListTasksByName:
    @staticmethod
    def _task(name, version):
        return SimpleNamespace(name=name, version=version)

    @pytest.fixture
    def listall(self, init_config, monkeypatch):
        listall = _ListAll([self._task("t", "v2"), self._task("t", "v1"), self._task("other", "v1")])
        monkeypatch.setattr(flyte.remote.Task, "listall", listall)
        return listall

    @pytest.mark.asyncio
    async def test_defaults_reproduce_the_previous_sdk_call(self, listall):
        await _fn(_env("task"), "list_tasks")()
        assert listall.calls == [
            {
                "project": "cfg-project",
                "domain": "cfg-domain",
                "sort_by": ("created_at", "desc"),
                "limit": 100,
                "entrypoint": None,
            }
        ]

    @pytest.mark.asyncio
    async def test_name_is_passed_as_by_task_name(self, listall):
        out = await _fn(_env("task"), "list_tasks")(name="t", limit=5)
        assert listall.calls[0]["by_task_name"] == "t"
        assert listall.calls[0]["limit"] == 5
        assert [(t["name"], t["version"]) for t in out] == [("t", "v2"), ("t", "v1"), ("other", "v1")]

    @pytest.mark.asyncio
    async def test_the_allowlist_still_filters_and_widens_the_scan(self, listall):
        env = _env("task", task_allowlist=["t"])
        out = await _fn(env, "list_tasks")(name="t", limit=5)
        assert listall.calls[0]["limit"] == ALLOWLIST_SCAN_LIMIT
        assert [(t["name"], t["version"]) for t in out] == [("t", "v2"), ("t", "v1")]


# ------------------------------
# registry
# ------------------------------


class TestRegistry:
    def test_create_trigger_is_a_plain_write(self):
        info = TOOL_REGISTRY["create_trigger"]
        assert (info.group, info.read_only, info.destructive, info.idempotent) == ("trigger", False, False, False)

    @pytest.mark.parametrize(("tool", "group"), [("delete_trigger", "trigger"), ("delete_app", "app")])
    def test_deletes_are_destructive_and_idempotent(self, tool, group):
        info = TOOL_REGISTRY[tool]
        assert (info.group, info.read_only, info.destructive, info.idempotent) == (group, False, True, True)

    @pytest.mark.parametrize("tool", ["create_trigger", "delete_trigger", "delete_app"])
    def test_read_only_servers_do_not_expose_the_new_tools(self, tool):
        env = FlyteMCPAppEnvironment(name="test-mcp", read_only=True)
        assert tool not in env.enabled_tools

    def test_annotations_reach_the_registered_tools(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        tools = env._mcp_server._tool_manager._tools
        assert tools["delete_app"].annotations.destructiveHint is True
        assert tools["delete_trigger"].annotations.idempotentHint is True
        assert tools["create_trigger"].annotations.readOnlyHint is False
        assert tools["create_trigger"].annotations.destructiveHint is False
