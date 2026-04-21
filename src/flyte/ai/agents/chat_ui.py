"""AgentChatAppEnvironment — FastAPI-based chat UI for any Agent."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import rich.repr
from pydantic import BaseModel

import flyte.app
from flyte.models import SerializationContext

from ._html import build_chat_html
from .protocol import Agent

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
    prompt_nudges:
        Optional list of prompt-nudge cards shown before the first
        message.  Each entry is a dict with ``"label"`` (short card
        title) and ``"prompt"`` (the query text sent when clicked).
    custom_css:
        Optional CSS string appended **after** the default styles.
        Use this to override colors, fonts, layout, etc. without
        replacing the entire stylesheet.  The default stylesheet is
        available as :data:`DEFAULT_CSS` for reference.
    logo_url:
        Optional URL to an image displayed to the left of the title
        in the header bar.  When ``None`` (default), no logo is shown.
    additional_buttons:
        Optional list of action-button dicts rendered to the right of
        the *Send* button.  Each dict must have ``"button_text"`` and
        ``"button_url"`` keys.  The first entry is displayed as a
        prominent primary button; any extra entries appear in a
        drop-up menu accessed via a chevron.
    """

    agent: Any = field(default=None)
    title: str | None = None
    prompt_nudges: list[dict[str, str]] = field(default_factory=list)
    custom_css: str = ""
    logo_url: str | None = None
    additional_buttons: list[dict[str, str]] = field(default_factory=list)
    type: str = "AgentChat"
    _caller_frame: inspect.FrameInfo | None = None

    def __post_init__(self):
        if self.agent is None:
            raise ValueError("'agent' is required for AgentChatAppEnvironment")

        if not isinstance(self.agent, Agent):
            raise TypeError(
                f"'agent' must implement the Agent protocol (run and tool_descriptions), got {type(self.agent)}"
            )

        super().__post_init__()
        self._server = self._fastapi_server

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            if caller_frame and caller_frame.f_back:
                self._caller_frame = inspect.getframeinfo(caller_frame.f_back)

    async def _fastapi_server(self):
        import time

        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse

        agent = self.agent
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
            result = await agent.run(req.message, req.history)
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
        chat_html = build_chat_html(
            title=display_title,
            custom_css=self.custom_css,
            logo_url=self.logo_url,
            additional_buttons=self.additional_buttons,
        )

        @fastapi_app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(content=chat_html)

        config = uvicorn.Config(fastapi_app, port=self.port.port)
        await uvicorn.Server(config).serve()

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []
