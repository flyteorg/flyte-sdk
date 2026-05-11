"""Tests for flyte.ai.chat.app."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from flyte.ai.agents import AgentResult
from flyte.ai.agents.protocol import Agent
from flyte.ai.chat import AgentChatAppEnvironment, CustomTheme
from flyte.ai.chat.app import _ChatRequest, _hex_to_rgb, _rgba
from flyte.app.extras import FastAPIPassthroughAuthMiddleware
from flyte.models import ActionPhase


class _StubAgent:
    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        return AgentResult(summary=f"echo:{message}")

    def tool_descriptions(self) -> list[dict[str, str]]:
        return [{"name": "n", "signature": "n()", "description": "d"}]


class TestHexHelpers:
    def test_hex_to_rgb_six_char(self):
        assert _hex_to_rgb("#aABBcc") == (170, 187, 204)

    def test_hex_to_rgb_three_char(self):
        assert _hex_to_rgb("#abc") == (170, 187, 204)

    def test_rgba_format(self):
        s = _rgba("#000000", 0.5)
        assert s == "rgba(0, 0, 0, 0.5)"


class TestCustomTheme:
    def test_valid_defaults(self):
        t = CustomTheme()
        css = t.to_css()
        assert "#6F2AEF" in css
        assert "rgba(" in css
        assert ".input-bar button" in css

    def test_custom_colors_in_css(self):
        t = CustomTheme(accent_color="#112233", accent_hover_color="#445566", button_text_color="#ffffff")
        css = t.to_css()
        assert "#112233" in css
        assert "#445566" in css
        assert "#ffffff" in css

    @pytest.mark.parametrize(
        "bad",
        ["red", "#gg0000", "#12", "#12345", ""],
    )
    def test_invalid_hex_raises(self, bad: str):
        with pytest.raises(ValueError, match="CSS hex"):
            CustomTheme(accent_color=bad)


class TestAgentChatAppEnvironment:
    def test_requires_agent(self):
        with pytest.raises(ValueError, match="'agent' is required"):
            AgentChatAppEnvironment(name="test-app", image="auto", agent=None)  # type: ignore[arg-type]

    def test_rejects_non_agent(self):
        class NotAnAgent:
            pass

        with pytest.raises(TypeError, match="Agent protocol"):
            AgentChatAppEnvironment(name="test-app", image="auto", agent=NotAnAgent())

    def test_accepts_protocol_agent(self):
        env = AgentChatAppEnvironment(name="test-app", image="auto", agent=_StubAgent())
        assert env.agent is not None
        assert isinstance(env.agent, Agent)

    @pytest.mark.asyncio
    async def test_chat_stream_returns_ndjson_done_line(self):
        env = AgentChatAppEnvironment(name="test-app", image="auto", agent=_StubAgent())
        app = env.build_fastapi_app()
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.stream(
            "POST",
            "/api/chat",
            json={"message": "hi", "history": [], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert "ndjson" in (resp.headers.get("content-type") or "").lower()
            body = b"".join(resp.iter_bytes())
        lines = [ln for ln in body.decode().split("\n") if ln.strip()]
        assert len(lines) >= 1
        last = json.loads(lines[-1])
        assert last["type"] == "done"
        assert last["summary"] == "echo:hi"
        assert last["elapsed_ms"] >= 0

    def test_container_command_empty(self):
        env = AgentChatAppEnvironment(name="test-app", image="auto", agent=_StubAgent())
        assert env.container_command(MagicMock()) == []

    def test_build_fastapi_app_passthrough_adds_middleware(self):
        env = AgentChatAppEnvironment(
            name="test-app",
            image="auto",
            agent=_StubAgent(),
            passthrough_auth=True,
        )
        app = env.build_fastapi_app()
        middleware_classes = [m.cls for m in app.user_middleware]
        assert FastAPIPassthroughAuthMiddleware in middleware_classes

    def test_build_fastapi_app_without_passthrough_skips_middleware(self):
        env = AgentChatAppEnvironment(
            name="test-app",
            image="auto",
            agent=_StubAgent(),
            passthrough_auth=False,
        )
        app = env.build_fastapi_app()
        middleware_classes = [m.cls for m in app.user_middleware]
        assert FastAPIPassthroughAuthMiddleware not in middleware_classes

    def test_passthrough_custom_excluded_paths(self):
        env = AgentChatAppEnvironment(
            name="test-app",
            image="auto",
            agent=_StubAgent(),
            passthrough_auth=True,
            passthrough_auth_excluded_paths=frozenset({"/health", "/api/chat"}),
        )
        app = env.build_fastapi_app()
        assert any(m.cls == FastAPIPassthroughAuthMiddleware for m in app.user_middleware)

    @pytest.mark.asyncio
    async def test_task_entrypoint_used_for_chat(self):
        # Ensure passthrough_auth is required when task_entrypoint is set.
        entry = MagicMock()
        env = AgentChatAppEnvironment(
            name="test-app",
            image="auto",
            agent=_StubAgent(),
            task_entrypoint=entry,
            passthrough_auth=True,
        )
        app = env.build_fastapi_app()
        route = next(r for r in app.routes if getattr(r, "path", None) == "/api/chat")

        class _FakeRun:
            def __init__(self, value: Any):
                self._value = value

                async def _wait(*args, **kwargs):
                    return None

                async def _outputs(*args, **kwargs):
                    return (self._value,)

                self.wait = MagicMock()
                self.wait.aio = _wait
                self.outputs = MagicMock()
                self.outputs.aio = _outputs

        async def _run(task, *args, **kwargs):
            # Return a dict shaped like AgentResult payload.
            return _FakeRun({"summary": f"task:{args[0]}", "attempts": 2})

        with patch("flyte.run", new=MagicMock(aio=_run)):
            out = await route.endpoint(_ChatRequest(message="hi", history=[{"role": "user", "content": "prev"}]))
        assert out.summary == "task:hi"
        assert out.attempts == 2

    @pytest.mark.asyncio
    async def test_task_entrypoint_one_arg(self):
        entry = MagicMock()

        # Signature introspection uses task_entrypoint.func when present.
        def func(message: str) -> str:
            return ""

        entry.func = func

        env = AgentChatAppEnvironment(
            name="test-app",
            image="auto",
            agent=_StubAgent(),
            task_entrypoint=entry,
            passthrough_auth=True,
        )
        app = env.build_fastapi_app()
        route = next(r for r in app.routes if getattr(r, "path", None) == "/api/chat")

        class _FakeRun:
            def __init__(self, value: Any):
                self._value = value

                async def _wait(*args, **kwargs):
                    return None

                async def _outputs(*args, **kwargs):
                    return (self._value,)

                self.wait = MagicMock()
                self.wait.aio = _wait
                self.outputs = MagicMock()
                self.outputs.aio = _outputs

        async def _run(task, *args, **kwargs):
            return _FakeRun({"summary": f"ok:{args[0]}", "attempts": 3})

        with patch("flyte.run", new=MagicMock(aio=_run)):
            out = await route.endpoint(_ChatRequest(message="yo", history=[]))
        assert out.summary == "ok:yo"
        assert out.attempts == 3

    @pytest.mark.asyncio
    async def test_task_entrypoint_failed_run_surfaces_error_in_response(self):
        entry = MagicMock()
        env = AgentChatAppEnvironment(
            name="test-app",
            image="auto",
            agent=_StubAgent(),
            task_entrypoint=entry,
            passthrough_auth=True,
        )
        app = env.build_fastapi_app()
        route = next(r for r in app.routes if getattr(r, "path", None) == "/api/chat")

        class _FakeRunFailed:
            phase = ActionPhase.FAILED

            async def _wait(*args, **kwargs):
                return None

            async def _details():
                ad = MagicMock()
                ei = MagicMock()
                ei.kind = "USER"
                ei.message = "task exploded"
                ad.error_info = ei
                ad.abort_info = None
                d = MagicMock()
                d.action_details = ad
                return d

            wait = MagicMock()
            wait.aio = _wait
            outputs = MagicMock()
            details = MagicMock()
            details.aio = _details

        async def _run(task, *args, **kwargs):
            return _FakeRunFailed()

        with patch("flyte.run", new=MagicMock(aio=_run)):
            out = await route.endpoint(_ChatRequest(message="hi", history=[]))
        assert "failed" in out.error.lower() or "USER" in out.error
        assert "task exploded" in out.error
        assert out.summary == ""

    @pytest.mark.asyncio
    async def test_task_entrypoint_outputs_error_surfaces_in_response(self):
        entry = MagicMock()
        env = AgentChatAppEnvironment(
            name="test-app",
            image="auto",
            agent=_StubAgent(),
            task_entrypoint=entry,
            passthrough_auth=True,
        )
        app = env.build_fastapi_app()
        route = next(r for r in app.routes if getattr(r, "path", None) == "/api/chat")

        class _FakeRunOkNoOutputs:
            phase = ActionPhase.SUCCEEDED

            async def _wait(*args, **kwargs):
                return None

            async def _outputs(*args, **kwargs):
                raise RuntimeError("no outputs on blob store")

            wait = MagicMock()
            wait.aio = _wait
            outputs = MagicMock()
            outputs.aio = _outputs

        async def _run(task, *args, **kwargs):
            return _FakeRunOkNoOutputs()

        with patch("flyte.run", new=MagicMock(aio=_run)):
            out = await route.endpoint(_ChatRequest(message="hi", history=[]))
        assert "outputs could not be loaded" in out.error
        assert "no outputs on blob store" in out.error

    @pytest.mark.asyncio
    async def test_task_entrypoint_nested_coroutine_is_fully_awaited(self):
        # Now that we use flyte.run, nested coroutine handling is validated
        # by treating outputs[0] as a coroutine.
        entry = MagicMock()
        env = AgentChatAppEnvironment(
            name="test-app",
            image="auto",
            agent=_StubAgent(),
            task_entrypoint=entry,
            passthrough_auth=True,
        )
        app = env.build_fastapi_app()
        route = next(r for r in app.routes if getattr(r, "path", None) == "/api/chat")

        async def inner(message: str) -> str:
            return f"inner:{message}"

        class _FakeRun:
            def __init__(self, value: Any):
                self._value = value

                async def _wait(*args, **kwargs):
                    return None

                async def _outputs(*args, **kwargs):
                    return (self._value,)

                self.wait = MagicMock()
                self.wait.aio = _wait
                self.outputs = MagicMock()
                self.outputs.aio = _outputs

        async def _run(task, *args, **kwargs):
            # outputs[0] is an un-awaited coroutine
            return _FakeRun(inner(args[0]))

        with patch("flyte.run", new=MagicMock(aio=_run)):
            out = await route.endpoint(_ChatRequest(message="zz", history=[]))
        assert out.summary == "inner:zz"
