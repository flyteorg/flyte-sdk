"""SGLang's HTTP app, served under the app's base path when its URL is path-based.

Requests reach the app with the path intact (/<prefix>/apps/<name>/v1/...), and
SGLang has neither a root path nor a middleware option. Its server's FastAPI app
is a module global: this adds ServeUnderBasePath to it before it serves.

With one HTTP worker, SGLang serves that app in the launching process. With more
(`--tokenizer-worker-num` > 1), uvicorn imports it afresh in each worker process,
from SGLANG_HTTP_APPS; the shim points it at `http_app` here instead, which adds
the middleware in the worker. It does not import the shim (torch, the model
loader), which the workers do not need.
"""

from __future__ import annotations

import importlib
import logging

from flyte.app._base_path import ServeUnderBasePath, app_base_path

logger = logging.getLogger(__name__)

# The modules SGLang's FastAPI app has lived in, newest first.
SGLANG_APP_MODULES = ("sglang.srt.entrypoints.http_server", "sglang.srt.server")

# The import strings SGLang gives uvicorn for it, with more than one HTTP worker.
SGLANG_HTTP_APPS = tuple(f"{module}:app" for module in SGLANG_APP_MODULES)

# The factory serving it under the base path in each of those workers.
HTTP_APP_FACTORY = f"{__name__}:http_app"


def serve_under_base_path():
    """Add ServeUnderBasePath to SGLang's HTTP app; the app, or None when there is no base path or no app."""
    base = app_base_path()
    if not base:
        return None
    app = None
    for module in SGLANG_APP_MODULES:
        try:
            app = getattr(importlib.import_module(module), "app", None)
        except ImportError:
            continue
        if app is not None:
            break
    if app is None or not hasattr(app, "add_middleware"):
        logger.warning(
            "This sglang version's HTTP app was not found, so it does not serve under the app's base path %s; "
            "requests under it will not be routed",
            base,
        )
        return None
    # Once per app: the launching process and a worker may both get here.
    if not any(getattr(m, "cls", None) is ServeUnderBasePath for m in getattr(app, "user_middleware", ())):
        logger.info("Serving under the app's base path %s", base)
        app.add_middleware(ServeUnderBasePath, base_path=base)
    return app


def http_app():
    """uvicorn's app factory for SGLang's HTTP workers: SGLang's app, under the base path."""
    app = serve_under_base_path()
    if app is None:
        app = importlib.import_module(SGLANG_APP_MODULES[0]).app
    return app
