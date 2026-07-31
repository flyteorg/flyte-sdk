"""Tests for the @nsys_profile decorator wiring and execute-wrapper branching."""

import asyncio
from unittest.mock import AsyncMock

import flyte
import pytest
from flyte._task import AsyncFunctionTaskTemplate

import flyteplugins.nsight._capture as capture
import flyteplugins.nsight._control as ctl
from flyteplugins.nsight import nsys_profile


class TestStaticWiring:
    def test_stamps_container_env_and_report(self):
        env = flyte.TaskEnvironment(name="test-env")

        @nsys_profile(trace=["cuda", "nvtx", "osrt"])
        @env.task
        async def t() -> str:
            return "ok"

        assert isinstance(t, AsyncFunctionTaskTemplate)
        assert t.env_vars[ctl.ENV_WRAPPER].startswith("nsys launch --session-new=")
        assert "--trace=cuda,nvtx,osrt" in t.env_vars[ctl.ENV_WRAPPER]
        assert t.env_vars[ctl.ENV_SESSION]
        assert t.env_vars[ctl.ENV_OUTPUT]
        # metrics need the deck, so it is enabled automatically
        assert t.report is True
        # execute is wrapped
        assert t.execute.__name__ == "wrapped_execute"

    def test_merges_existing_env_vars(self):
        env = flyte.TaskEnvironment(name="test-env")

        @env.task
        async def base() -> str:
            return "ok"

        base = base.override(env_vars={"MY_VAR": "1"})
        decorated = nsys_profile(trace=["cuda"])(base)
        assert decorated.env_vars["MY_VAR"] == "1"
        assert ctl.ENV_WRAPPER in decorated.env_vars

    def test_clustered_task_gets_opt_in(self):
        # A ClusteredTaskEnvironment task (task_type "clustered-task") opts the runtime into wrapping
        # its primary worker under nsys; a plain task does not.
        from flyte.clustered import ClusteredTaskEnvironment

        cenv = ClusteredTaskEnvironment(name="c-env", replicas=1, nproc_per_node=2)

        @nsys_profile(trace=["cuda", "nvtx"])
        @cenv.task
        async def clustered() -> str:
            return "ok"

        assert clustered.env_vars[ctl.ENV_CLUSTERED] == "1"

        plain_env = flyte.TaskEnvironment(name="plain")

        @nsys_profile(trace=["cuda"])
        @plain_env.task
        async def plain() -> str:
            return "ok"

        assert ctl.ENV_CLUSTERED not in plain.env_vars

    def test_enabled_false_is_passthrough(self):
        env = flyte.TaskEnvironment(name="test-env")

        @env.task
        async def base() -> str:
            return "ok"

        decorated = nsys_profile(enabled=False)(base)
        # unchanged: same object, no wrapper env stamped
        assert decorated is base
        assert not (base.env_vars or {}).get(ctl.ENV_WRAPPER)

    def test_rejects_non_task(self):
        with pytest.raises(TypeError):

            @nsys_profile()
            async def not_a_task():
                return "ok"

    def test_rejects_bad_capture(self):
        env = flyte.TaskEnvironment(name="test-env")

        @env.task
        async def base() -> str:
            return "ok"

        with pytest.raises(ValueError):
            nsys_profile(capture="everything")(base)


class TestExecuteBranching:
    """Exercise the wrapped execute without a GPU by stubbing the base execute and nsys control."""

    def _decorated(self, monkeypatch, **kwargs):
        # Patch the base class execute so original_execute (captured post-override) is a stub
        # that needs no task context.
        async def stub_execute(self, *a, **k):
            return "BODY"

        monkeypatch.setattr(AsyncFunctionTaskTemplate, "execute", stub_execute)
        env = flyte.TaskEnvironment(name="test-env")

        @nsys_profile(**kwargs)
        @env.task
        async def t() -> str:
            return "BODY"

        return t

    def test_passthrough_when_not_under_nsys(self, monkeypatch):
        t = self._decorated(monkeypatch)
        monkeypatch.setattr(ctl, "under_nsys", lambda: False)
        monkeypatch.setattr(ctl, "nsys_available", lambda: True)
        start = AsyncMock()
        monkeypatch.setattr(ctl, "start_collection", start)

        assert asyncio.run(t.execute()) == "BODY"
        start.assert_not_awaited()  # no collection attempted off-nsys

    def test_profiles_when_under_nsys(self, monkeypatch):
        t = self._decorated(monkeypatch)
        monkeypatch.setattr(ctl, "under_nsys", lambda: True)
        start = AsyncMock(return_value="/tmp/nsys/a0/report.nsys-rep")
        stop = AsyncMock()
        finalize = AsyncMock(return_value={"kernel_launches": 5})
        monkeypatch.setattr(ctl, "start_collection", start)
        monkeypatch.setattr(ctl, "stop_collection", stop)
        monkeypatch.setattr(capture, "finalize", finalize)

        assert asyncio.run(t.execute()) == "BODY"
        start.assert_awaited_once()
        stop.assert_awaited_once()
        finalize.assert_awaited_once()

    def test_finalizes_even_when_body_raises(self, monkeypatch):
        async def boom_execute(self, *a, **k):
            raise ValueError("kaboom")

        monkeypatch.setattr(AsyncFunctionTaskTemplate, "execute", boom_execute)
        env = flyte.TaskEnvironment(name="test-env")

        @nsys_profile()
        @env.task
        async def t() -> str:
            return "unused"

        monkeypatch.setattr(ctl, "under_nsys", lambda: True)
        monkeypatch.setattr(ctl, "start_collection", AsyncMock(return_value="/tmp/x.nsys-rep"))
        stop = AsyncMock()
        finalize = AsyncMock(return_value={})
        monkeypatch.setattr(ctl, "stop_collection", stop)
        monkeypatch.setattr(capture, "finalize", finalize)

        with pytest.raises(ValueError, match="kaboom"):
            asyncio.run(t.execute())
        # a profile of the failure is still finalized
        stop.assert_awaited_once()
        finalize.assert_awaited_once()

    def test_manual_capture_skips_auto_collection(self, monkeypatch):
        t = self._decorated(monkeypatch, capture="manual")
        monkeypatch.setattr(ctl, "under_nsys", lambda: True)
        start = AsyncMock()
        monkeypatch.setattr(ctl, "start_collection", start)

        assert asyncio.run(t.execute()) == "BODY"
        start.assert_not_awaited()  # manual mode leaves collection to nsys.range(...)
