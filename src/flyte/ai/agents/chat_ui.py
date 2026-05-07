"""AgentChatAppEnvironment — FastAPI-based chat UI for any Agent."""

from __future__ import annotations

import inspect
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import rich.repr
from pydantic import BaseModel

import flyte.app
from flyte.models import SerializationContext

from ._css import CUSTOM_THEME_CSS_TEMPLATE
from ._html import build_chat_html
from .protocol import Agent, AgentResult

# ------------------------------------------------------------------
# CustomTheme — human-readable color theming for the chat UI
# ------------------------------------------------------------------

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


@dataclass(kw_only=True)
class CustomTheme:
    """Declarative color theme for the Agent Chat UI.

    All colors should be CSS hex strings (e.g. ``"#E6A71F"``).

    Parameters
    ----------
    accent_color:
        Primary brand color used for links, highlights, active
        indicators, and solid-background buttons.  Defaults to the
        built-in purple (``"#6F2AEF"``).
    accent_hover_color:
        Lighter variant shown on hover states for accent-colored
        elements.  Defaults to ``"#8B52F2"``.
    button_text_color:
        Text color rendered *on top of* accent-colored buttons.
        Should contrast well with *accent_color*.  Defaults to
        ``"#f3f4f6"`` (near-white).
    """

    accent_color: str = "#6F2AEF"
    accent_hover_color: str = "#8B52F2"
    button_text_color: str = "#f3f4f6"

    def __post_init__(self) -> None:
        for attr in ("accent_color", "accent_hover_color", "button_text_color"):
            val = getattr(self, attr)
            if not _HEX_RE.match(val):
                raise ValueError(f"CustomTheme.{attr} must be a CSS hex color (e.g. '#E6A71F'), got {val!r}")

    def to_css(self) -> str:
        """Generate a CSS override string from the theme colors."""
        ac = self.accent_color
        ah = self.accent_hover_color
        bt = self.button_text_color
        return CUSTOM_THEME_CSS_TEMPLATE.format(
            ac=ac,
            ah=ah,
            bt=bt,
            ac_rgba_085=_rgba(ac, 0.85),
            ac_rgba_02=_rgba(ac, 0.2),
            ac_rgba_008=_rgba(ac, 0.08),
            ac_rgba_045=_rgba(ac, 0.45),
            ac_rgba_006=_rgba(ac, 0.06),
            ac_rgba_022=_rgba(ac, 0.22),
            ac_rgba_004=_rgba(ac, 0.04),
            ac_rgba_018=_rgba(ac, 0.18),
            ac_rgba_038=_rgba(ac, 0.38),
            ac_rgba_05=_rgba(ac, 0.5),
            ah_rgba_085=_rgba(ah, 0.85),
            ac_rgba_010=_rgba(ac, 0.10),
            ac_rgba_012=_rgba(ac, 0.12),
            ac_rgba_03=_rgba(ac, 0.3),
            ac_rgba_025=_rgba(ac, 0.25),
        )


# ------------------------------------------------------------------
# Request / response models (module-level for FastAPI schema compat)
# ------------------------------------------------------------------


class _ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class _ChatResponse(BaseModel):
    code: str = ""
    charts: list[str] = []
    summary: str = ""
    error: str = ""
    elapsed_ms: int = 0
    attempts: int = 1


async def _task_run_error_message(run_handle: Any) -> str:
    """Human-readable explanation when a remote :class:`~flyte.remote.Run` ends unsuccessfully."""
    phase = getattr(run_handle, "phase", "unknown")
    parts: list[str] = [f"Task run ended in state {phase}."]
    details_aio = getattr(getattr(run_handle, "details", None), "aio", None)
    if not callable(details_aio):
        return " ".join(parts)
    try:
        details = await details_aio()
        ad = getattr(details, "action_details", None)
        if ad is None:
            return " ".join(parts)
        err = getattr(ad, "error_info", None)
        if err is not None:
            kind = getattr(err, "kind", "error")
            message = getattr(err, "message", "") or ""
            parts.append(f"{kind}: {message}".strip())
        abort = getattr(ad, "abort_info", None)
        if err is None and abort is not None:
            reason = getattr(abort, "reason", "") or str(abort)
            parts.append(f"Aborted: {reason}".strip())
    except Exception as e:
        parts.append(f"(Could not load error details: {e})")
    return " ".join(parts)


# ------------------------------------------------------------------
# AgentChatAppEnvironment
# ------------------------------------------------------------------


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class AgentChatAppEnvironment(flyte.app.AppEnvironment):
    """An :class:`~flyte.app.AppEnvironment` that spins up a FastAPI chat
    interface backed by any object satisfying the :class:`Agent` protocol.

    Parameters
    ----------
    agent:
        Any object implementing the :class:`Agent` protocol.
    title:
        Title displayed in the UI header and browser tab. Defaults to
        the environment *name*.
    subtitle:
        Optional short subtitle displayed below the title in the
        header area.  Use it to explain what the agent does.
    prompt_nudges:
        Optional list of prompt-nudge cards shown before the first
        message.  Each entry is a dict with ``"label"`` (short card
        title) and ``"prompt"`` (the query text sent when clicked).
    theme:
        Optional :class:`CustomTheme` instance that controls the UI
        accent colors via human-readable attributes.  When provided,
        the theme CSS is generated automatically and prepended to any
        *custom_css*.
    custom_css:
        Optional CSS string appended **after** the default styles
        (and after theme CSS, if a *theme* is provided).  Use this
        for fine-grained overrides beyond what :class:`CustomTheme`
        exposes.
    logo_url:
        Optional URL to an image displayed to the left of the title
        in the header bar.  When ``None`` (default), no logo is shown.
    additional_buttons:
        Optional list of action-button dicts rendered to the right of
        the *Send* button.  Each dict must have ``"button_text"`` and
        ``"button_url"`` keys.  The first entry is displayed as a
        prominent primary button; any extra entries appear in a
        drop-up menu accessed via a chevron.
    passthrough_auth:
        When ``True``, the FastAPI app initializes ``flyte.init_passthrough`` at
        startup and adds ``FastAPIPassthroughAuthMiddleware`` so incoming
        ``Authorization`` / cookie headers are forwarded to Flyte remote calls.
        Enable this when using ``CodeModeAgent`` with ``@env.task`` tools —
        nested task execution needs caller credentials (same pattern as
        ``FlyteWebhookAppEnvironment``).
    passthrough_auth_excluded_paths:
        Paths skipped by passthrough middleware (defaults to common docs and
        health routes). Only used when ``passthrough_auth`` is ``True``.
    task_entrypoint:
        Optional Flyte task used as the chat handler entrypoint.

        When set, ``/api/chat`` calls the task (via ``task_entrypoint.aio``)
        instead of calling ``agent.run`` directly. This is useful for agents
        whose tool calls must run under a parent task context (e.g. a
        ``CodeModeAgent`` using durable ``@env.task`` tools).

        The entrypoint may accept either:

        - ``(message: str, history: list[dict[str, str]])``; or
        - ``(message: str)``.

        The return value may be an :class:`~flyte.ai.agents.protocol.AgentResult`,
        a dict with keys like ``summary``/``charts``/``code``, or a plain string
        (treated as ``summary``).
    """

    agent: Any = field(default=None)
    title: str | None = None
    subtitle: str | None = None
    prompt_nudges: list[dict[str, str]] = field(default_factory=list)
    theme: CustomTheme | None = None
    custom_css: str = ""
    logo_url: str | None = None
    additional_buttons: list[dict[str, str]] = field(default_factory=list)
    passthrough_auth: bool = False
    passthrough_auth_excluded_paths: frozenset[str] | None = None
    task_entrypoint: Any | None = None
    type: str = "AgentChat"
    _caller_frame: inspect.FrameInfo | None = None

    def __post_init__(self):
        if self.agent is None:
            raise ValueError("'agent' is required for AgentChatAppEnvironment")

        if not isinstance(self.agent, Agent):
            raise TypeError(
                f"'agent' must implement the Agent protocol (run and tool_descriptions), got {type(self.agent)}"
            )

        if self.task_entrypoint is not None and not self.passthrough_auth:
            raise ValueError(
                "task_entrypoint requires passthrough_auth=True so the app can run tasks with caller credentials."
            )

        super().__post_init__()
        self._server = self._fastapi_server

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            if caller_frame and caller_frame.f_back:
                self._caller_frame = inspect.getframeinfo(caller_frame.f_back)

    def build_fastapi_app(self) -> Any:
        """Construct the FastAPI application (routes, HTML shell, optional auth).

        Useful for tests and advanced mounting; the deployed server uses this via
        :meth:`_fastapi_server`.
        """
        import time

        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse

        agent = self.agent
        task_entrypoint = self.task_entrypoint

        if self.passthrough_auth:
            import flyte
            from flyte.app.extras import FastAPIPassthroughAuthMiddleware

            @asynccontextmanager
            async def lifespan(app: FastAPI):
                await flyte.init_passthrough.aio(
                    project=flyte.current_project(),
                    domain=flyte.current_domain(),
                )
                yield

            fastapi_app = FastAPI(title=self.title or self.name, lifespan=lifespan)
            excluded = self.passthrough_auth_excluded_paths or frozenset(
                {"/health", "/docs", "/openapi.json", "/redoc"}
            )
            fastapi_app.add_middleware(FastAPIPassthroughAuthMiddleware, excluded_paths=set(excluded))
        else:
            fastapi_app = FastAPI(title=self.title or self.name)

        @fastapi_app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        @fastapi_app.get("/api/tools")
        async def get_tools() -> JSONResponse:
            return JSONResponse(content=agent.tool_descriptions())

        nudges = self.prompt_nudges

        @fastapi_app.get("/api/nudges")
        async def get_nudges() -> JSONResponse:
            return JSONResponse(content=nudges)

        @fastapi_app.post("/api/chat")
        async def chat(req: _ChatRequest) -> _ChatResponse:
            t0 = time.monotonic()
            if task_entrypoint is None:
                result_obj: Any = await agent.run(req.message, req.history)
            else:
                import flyte
                from flyte.models import ActionPhase

                # Prefer introspecting the underlying Python function if present
                fn = getattr(task_entrypoint, "func", None)
                n_params = 1
                if fn is not None:
                    try:
                        n_params = len(inspect.signature(fn).parameters)
                    except Exception:
                        n_params = 1

                try:
                    if n_params >= 2:
                        run_handle = await flyte.run.aio(task_entrypoint, req.message, req.history)
                    else:
                        run_handle = await flyte.run.aio(task_entrypoint, req.message)

                    # In remote/hybrid mode flyte.run returns a Run; wait + fetch outputs.
                    if hasattr(run_handle, "wait") and hasattr(run_handle, "outputs"):
                        await run_handle.wait.aio(quiet=True)
                        phase = getattr(run_handle, "phase", ActionPhase.SUCCEEDED)
                        if phase != ActionPhase.SUCCEEDED:
                            result_obj = AgentResult(
                                summary="",
                                error=await _task_run_error_message(run_handle),
                            )
                        else:
                            try:
                                outs = await run_handle.outputs.aio()
                            except Exception as e:
                                result_obj = AgentResult(
                                    summary="",
                                    error=f"Task succeeded but outputs could not be loaded: {e}",
                                )
                            else:
                                result_obj = outs[0] if len(outs) > 0 else None
                    else:
                        # Local mode may return the raw result directly
                        result_obj = run_handle
                except Exception as e:
                    result_obj = AgentResult(summary="", error=f"Task run failed: {e}")

            # In local mode, Flyte task `.aio()` may yield an un-awaited coroutine
            # (e.g. from forward() returning a coroutine for async functions).
            while inspect.iscoroutine(result_obj):
                result_obj = await result_obj

            if isinstance(result_obj, AgentResult):
                result = result_obj
            elif isinstance(result_obj, dict):
                result = AgentResult(
                    code=str(result_obj.get("code", "")),
                    charts=list(result_obj.get("charts", [])) if "charts" in result_obj else [],
                    summary=str(result_obj.get("summary", "")),
                    error=str(result_obj.get("error", "")),
                    attempts=int(result_obj.get("attempts", 1)),
                )
            elif isinstance(result_obj, str):
                result = AgentResult(summary=result_obj)
            else:
                result = AgentResult(summary=str(result_obj))
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return _ChatResponse(
                code=result.code,
                charts=result.charts,
                summary=result.summary,
                error=result.error,
                elapsed_ms=elapsed_ms,
                attempts=result.attempts,
            )

        display_title = self.title or self.name
        css_parts: list[str] = []
        if self.theme is not None:
            css_parts.append(self.theme.to_css())
        if self.custom_css:
            css_parts.append(self.custom_css)
        chat_html = build_chat_html(
            title=display_title,
            custom_css="\n".join(css_parts),
            logo_url=self.logo_url,
            additional_buttons=self.additional_buttons,
            subtitle=self.subtitle,
        )

        @fastapi_app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(content=chat_html)

        return fastapi_app

    async def _fastapi_server(self):
        import uvicorn

        fastapi_app = self.build_fastapi_app()
        config = uvicorn.Config(fastapi_app, port=self.port.port)
        await uvicorn.Server(config).serve()

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []
