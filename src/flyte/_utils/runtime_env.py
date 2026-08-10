"""
Runtime-capability detection for constrained execution environments.

The primary motivating environment is Pyodide / WebAssembly (running flyte in the browser),
where the host Python interpreter has **no thread support**. This module centralizes the
detection so the rest of the SDK can opt into a thread-free, async-only execution mode.
"""

import os
import sys

from flyte._utils.helpers import str2bool

# True when running under Pyodide / Emscripten (browser WASM). The CPython build there reports
# ``sys.platform == "emscripten"`` and cannot start OS threads.
IS_PYODIDE = sys.platform == "emscripten"


def background_loop_disabled() -> bool:
    """
    Whether the syncify background event-loop thread must be skipped.

    When True, flyte runs in an async-only mode: the synchronous (blocking) flyte API is
    unavailable and callers must use the ``.aio()`` coroutine form, which is executed directly
    in the caller's own running event loop instead of a dedicated background thread.

    Controlled by ``FLYTE_DISABLE_BACKGROUND_LOOP`` (truthy/falsy string); when unset it
    auto-enables under Pyodide so the browser case works without any configuration.
    """
    env = os.getenv("FLYTE_DISABLE_BACKGROUND_LOOP")
    if env not in (None, ""):
        return str2bool(env)
    return IS_PYODIDE
