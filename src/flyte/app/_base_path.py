"""The path an app is served under, when its URL is path-based.

A platform can serve apps under a path of a host it shares with other apps
(self-hosted onebox at https://platform.example.com/onebox/apps/<name>/). It
then tells the app that path in `_U_APP_BASE_PATH`, and requests reach the
app with the path intact.
"""

from __future__ import annotations

import os
import typing

APP_BASE_PATH_ENV_VAR = "_U_APP_BASE_PATH"

# Streamlit reads every option from STREAMLIT_<SECTION>_<OPTION>: this is
# --server.baseUrlPath. A --server.baseUrlPath on the command line still wins.
STREAMLIT_BASE_URL_PATH_ENV_VAR = "STREAMLIT_SERVER_BASE_URL_PATH"

# Gradio's root_path: it serves under it, whether or not the proxy removed it,
# and builds its URLs with it. A root_path passed to launch() still wins.
GRADIO_ROOT_PATH_ENV_VAR = "GRADIO_ROOT_PATH"

# The import path of ServeUnderBasePath, for servers that take ASGI middleware
# by name (vLLM's --middleware).
SERVE_UNDER_BASE_PATH_MIDDLEWARE = f"{__name__}.ServeUnderBasePath"


def app_base_path(environ: typing.Mapping[str, str] = os.environ) -> str:
    """The app's base path without its trailing slash, or "" when it is served at its host's root."""
    return environ.get(APP_BASE_PATH_ENV_VAR, "").rstrip("/")


def apply_base_path_defaults(environ: typing.MutableMapping[str, str] = os.environ) -> None:
    """Point the app servers that read their base path from the environment at the app's.

    Without it, Streamlit serves at / and answers the app's own path with a
    redirect that loops against the platform's, and Gradio answers it with a
    404. A value the app sets itself is kept.
    """
    base = app_base_path(environ)
    if base:
        environ.setdefault(STREAMLIT_BASE_URL_PATH_ENV_VAR, base)
        environ.setdefault(GRADIO_ROOT_PATH_ENV_VAR, base)


class ServeUnderBasePath:
    """ASGI middleware: serves an ASGI app under a base path, whether or not the proxy in front removed it.

    ASGI's root_path is the path the app is mounted under, and a request's path includes it;
    Starlette (and FastAPI) route on the path below root_path and build URLs (url_for, the
    docs' openapi.json) with it. A request that arrives without the base path, from a proxy
    that removed it, gets it put back, so routing and generated URLs agree either way.
    (uvicorn's --root-path instead assumes the proxy removed it, and prepends it again.)

    The base path defaults to the app's (`_U_APP_BASE_PATH`); with none, requests pass
    through unchanged. That is how servers that take middleware by import path
    construct it (`ServeUnderBasePath(app)`).
    """

    def __init__(self, app, base_path: str | None = None):
        self.app = app
        self.base_path = (app_base_path() if base_path is None else base_path).rstrip("/")

    async def __call__(self, scope, receive, send):
        if self.base_path and scope["type"] in ("http", "websocket"):
            scope = dict(scope)
            path = scope.get("path", "")
            if not (path == self.base_path or path.startswith(self.base_path + "/")):
                scope["path"] = self.base_path + path
                if scope.get("raw_path") is not None:
                    scope["raw_path"] = self.base_path.encode() + scope["raw_path"]
            scope["root_path"] = self.base_path
        await self.app(scope, receive, send)
