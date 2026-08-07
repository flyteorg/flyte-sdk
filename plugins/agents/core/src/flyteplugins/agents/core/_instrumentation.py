"""Letting an out-of-tree observability package instrument the frameworks adapters drive.

Instrumentation libraries for agent frameworks generally work by being handed something at
the call site: a callback list for LangChain, a ``RunHooks`` for OpenAI Agents, options for
the Claude Agent SDK. That is a problem here because the adapter owns the call. A user who
wants their agent instrumented has nowhere to put the handler since the invocation happens
inside our code rather than theirs.

So adapters offer the framework-native payload here on the way past, and anything registered
gets a chance to return a modified one. ``flyteplugins-agento11y`` is the first registrant.
This module deliberately knows nothing about any observability vendor: it moves an opaque
object through a function and keeps the agents plugin free of those dependencies.

Adapters call `apply_instrumentation` at the invocation site:

    config = apply_instrumentation("langgraph", config)
    result = await agent.ainvoke(state, config=config)

With nothing registered this returns the payload unchanged, so the ordinary path is a dict
lookup and nothing else.
"""

from __future__ import annotations

import typing

from flyte._logging import logger

__all__ = [
    "CallWrapper",
    "Instrumentor",
    "apply_call_wrapper",
    "apply_instrumentation",
    "instrumented_frameworks",
    "register_call_wrapper",
    "register_instrumentor",
    "unregister_call_wrapper",
    "unregister_instrumentor",
]

# Takes the framework's native run payload and returns one with instrumentation attached.
# The type is intentionally loose: what flows through is whatever the framework expects,
# a config dict for LangChain, a ClaudeAgentOptions for the Claude Agent SDK, and so on.
Instrumentor = typing.Callable[[typing.Any], typing.Any]

_instrumentors: dict[str, Instrumentor] = {}


def register_instrumentor(framework: str, instrumentor: Instrumentor) -> None:
    """Register the instrumentor for a framework, replacing any previous one.

    Args:
        framework: Adapter name, matching the plugin directory: "langgraph", "langchain",
            "openai", "claude", "google", "pydantic_ai".
        instrumentor: Called with the framework's run payload, returns the modified payload.
    """
    _instrumentors[framework] = instrumentor


def unregister_instrumentor(framework: str) -> None:
    """Remove a framework's instrumentor, if one is registered."""
    _instrumentors.pop(framework, None)


def instrumented_frameworks() -> tuple[str, ...]:
    """Frameworks with an instrumentor registered, for diagnostics and tests."""
    return tuple(sorted(_instrumentors))


def apply_instrumentation(framework: str, payload: typing.Any) -> typing.Any:
    """Offer a framework's run payload to its instrumentor and return the result.

    Returns the payload untouched when nothing is registered for the framework, and also when
    the instrumentor fails: instrumentation is never a reason to fail the agent it is
    watching, and a half-modified payload is worse than an uninstrumented one.
    """
    instrumentor = _instrumentors.get(framework)
    if instrumentor is None:
        return payload
    try:
        return instrumentor(payload)
    except Exception:
        logger.debug(f"Instrumentor for {framework!r} failed; running uninstrumented", exc_info=True)
        return payload


# Some SDKs cannot be instrumented by handing them an object at the call site. The Claude
# Agent SDK is the case in point: its model turns arrive as messages on the stream returned by
# ``query``, so recording them means wrapping that call rather than decorating its arguments.
# A wrapper takes the callable the adapter would have used and returns one to use instead.
CallWrapper = typing.Callable[[typing.Any], typing.Any]

_call_wrappers: dict[str, CallWrapper] = {}


def register_call_wrapper(framework: str, wrapper: CallWrapper) -> None:
    """Register the call wrapper for a framework, replacing any previous one."""
    _call_wrappers[framework] = wrapper


def unregister_call_wrapper(framework: str) -> None:
    """Remove a framework's call wrapper, if one is registered."""
    _call_wrappers.pop(framework, None)


def apply_call_wrapper(framework: str, call: typing.Any) -> typing.Any:
    """Offer a framework's call to its wrapper and return whatever should be called instead.

    Returns the original callable when nothing is registered, and also when the wrapper fails:
    a half-wrapped call is worse than an uninstrumented one, and observing an agent must never
    be the reason it stops working.
    """
    wrapper = _call_wrappers.get(framework)
    if wrapper is None:
        return call
    try:
        return wrapper(call)
    except Exception:
        logger.debug(f"Call wrapper for {framework!r} failed; calling uninstrumented", exc_info=True)
        return call
