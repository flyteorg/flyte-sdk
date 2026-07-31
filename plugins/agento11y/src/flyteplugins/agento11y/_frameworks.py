"""Registering agento11y's framework handlers with the agents plugin.

Every agento11y integration exposes the same shape: a ``with_agento11y_*`` function that
takes the framework's native run payload and returns it with a handler attached. That lines
up exactly with the instrumentor contract in ``flyteplugins.agents.core``, so wiring the two
together is a table rather than six bespoke integrations.

The reason this indirection exists at all is that the adapters own the framework call. A
LangChain callback or an OpenAI Agents ``RunHooks`` has to be passed at invocation time, and
a user driving an agent through ``run_agent`` has nowhere to put one. The adapter offers its
payload to the registry on the way past and this module is what answers.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from flyte._logging import logger

__all__ = ["SUPPORTED_FRAMEWORKS", "register_all", "unregister_all"]


@dataclass(frozen=True)
class _Integration:
    """Where a framework's injector lives and what it accepts.

    ``module`` is the import name rather than the distribution name; they differ for the
    Claude Agent SDK, whose ``agento11y-claude-agent-sdk`` package imports as
    ``agento11y_claude_agent``.

    ``async_handler`` records whether the injector takes that keyword. Passing it where it is
    not accepted would be swallowed by ``**handler_kwargs`` and forwarded to the handler
    constructor, which is a quieter and more confusing failure than not passing it.
    """

    framework: str
    module: str
    function: str
    async_handler: bool
    # Whether the framework's payload is a config dict carrying a ``metadata`` mapping. The
    # shared handler reads a conversation id out of metadata, which is the only way to give
    # framework-driven generations the Flyte run as their conversation: unlike agent name,
    # the handler resolves conversation_id itself and so never falls back to our contextvar.
    conversation_via_metadata: bool = False
    # Whether the framework's payload carries a ``context`` object the handler reads a
    # conversation id from. OpenAI Agents passes the caller's context through to its hooks,
    # which is the only slot in that framework where a conversation id can be supplied.
    conversation_via_context: bool = False
    # Whether the adapter offers its whole run-kwargs dict rather than the injector's payload
    # alone. Pydantic AI does, because ``Agent.run`` takes a native conversation_id alongside
    # the capabilities list and an instrumentor reaching only the list could never set it.
    payload_is_run_kwargs: str = ""
    # Whether this framework is instrumented by wrapping the adapter's call rather than by
    # decorating a payload. The Claude Agent SDK reports model turns as messages on the stream
    # returned by ``query``, so only a wrapper around that call can record generations; its
    # options-based hooks cover tool calls alone.
    call_wrapper: str = ""


# The adapters we drive that agento11y has an integration for. crewai and mistral have
# adapters here but no agento11y package, so they are absent rather than broken.
SUPPORTED_FRAMEWORKS: tuple[_Integration, ...] = (
    _Integration("langchain", "agento11y_langchain", "with_agento11y_langchain_callbacks", True, True),
    _Integration("langgraph", "agento11y_langgraph", "with_agento11y_langgraph_callbacks", True, True),
    _Integration("openai", "agento11y_openai_agents", "with_agento11y_openai_agents_hooks", True,
                 conversation_via_context=True),
    _Integration("claude", "agento11y_claude_agent", "agento11y_query", False, call_wrapper="agento11y_query"),
    _Integration("google", "agento11y_google_adk", "with_agento11y_google_adk_plugins", True),
    _Integration("pydantic_ai", "agento11y_pydantic_ai", "with_agento11y_pydantic_ai_capability", False,
                 conversation_via_metadata=True, payload_is_run_kwargs="capabilities"),
)


def _load(integration: _Integration) -> typing.Any | None:
    """Import a framework's injector, or return None when its package is not installed.

    Absence is the normal case: nobody installs all six. It is reported at debug level so an
    unexpected absence is still discoverable without making the common case noisy.
    """
    try:
        module = __import__(integration.module, fromlist=[integration.function])
        return getattr(module, integration.function)
    except Exception:
        logger.debug(
            f"agento11y integration for {integration.framework!r} unavailable "
            f"(install flyteplugins-agento11y[{integration.framework.replace('_', '-')}])",
        )
        return None


def _with_conversation_metadata(config: typing.Any) -> typing.Any:
    """Put the Flyte run name into a runnable config's metadata as the conversation id.

    :class:`~flyteplugins.agento11y.FlyteIdentityBinding` sets a conversation contextvar, and
    the client honours it for generations recorded directly. Framework handlers do not: they
    resolve a conversation id from the callback payload themselves, so the contextvar is never
    consulted and the run's generations end up ungrouped. Metadata is the payload key the
    handler looks in, so this is where the run name has to go.

    Anything the caller already set wins, since an explicit conversation id means the run is
    not the unit of conversation for them.
    """
    run_name = _current_run_name()
    if not run_name:
        return config

    merged = dict(config or {})
    existing = merged.get("metadata")
    if callable(existing):
        # Pydantic AI accepts a metadata factory. Wrapping it to add a key would change what
        # the caller's function returns, so a factory is left alone.
        return merged
    metadata = dict(existing or {})
    metadata.setdefault("conversation_id", run_name)
    merged["metadata"] = metadata
    return merged


def _with_conversation_context(run_options: typing.Any) -> typing.Any:
    """Put the Flyte run name into an OpenAI Agents run context as the conversation id.

    OpenAI Agents has no config dict to carry metadata; its integration reads the conversation
    id off the run context object instead, which the SDK threads through to its hooks. Without
    this the handler falls back to a synthesized ``agento11y:framework:...`` id and the run's
    generations are not grouped by the Flyte run.

    A caller-supplied context is left alone: it belongs to their tools, and overwriting it to
    win a grouping label would be a poor trade.
    """
    merged = dict(run_options or {})
    if merged.get("context") is not None:
        return merged

    run_name = _current_run_name()
    if not run_name:
        return merged

    merged["context"] = {"conversation_id": run_name}
    return merged


def _current_run_name() -> str | None:
    """The run this task belongs to, or None outside a task context."""
    try:
        import flyte

        ctx = flyte.ctx()
        return ctx.action.run_name if ctx else None
    except Exception:
        logger.debug("Could not read the Flyte run name for the conversation id", exc_info=True)
        return None


def _instrument_run_kwargs(
    run_kwargs: typing.Any,
    injector: typing.Any,
    integration: _Integration,
    client: typing.Any,
) -> typing.Any:
    """Attach the handler and the conversation id to a framework's whole run payload.

    Used where the framework names a conversation itself, so the run can be bound directly
    instead of going through a metadata slot. A caller-supplied conversation id wins.
    """
    merged = dict(run_kwargs or {})
    key = integration.payload_is_run_kwargs
    merged[key] = injector(merged.get(key), client=client)

    run_name = _current_run_name()
    if run_name and not merged.get("conversation_id"):
        # Pydantic AI's own telemetry uses this, and it is stamped onto ModelMessage, so the
        # two systems agree on the conversation even though agento11y ignores it.
        merged["conversation_id"] = run_name
    if integration.conversation_via_metadata:
        # agento11y's capability never reads RunContext.conversation_id; it reads
        # RunContext.metadata. Without this the capability synthesizes its own id from
        # ctx.run_id and the run's generations are not grouped by the Flyte run.
        merged = _with_conversation_metadata(merged)
    return merged


def _make_call_wrapper(wrapped_query: typing.Any, client: typing.Any) -> typing.Any:
    """Build a replacement for the SDK's query that records the stream it yields.

    agento11y's wrapper takes the underlying query as ``_query_fn``, drives the same stream,
    and records each message, so the adapter's own loop over the messages is unaffected.
    """

    def wrap(call: typing.Any) -> typing.Any:
        def instrumented(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            handler_kwargs: dict[str, typing.Any] = {}
            # This handler takes a conversation id directly and prefers it over the SDK's
            # session id, so unlike the payload-based frameworks the run can be bound without
            # going through a metadata slot. Resolved here rather than at registration because
            # the wrapper is built once and called inside each task.
            run_name = _current_run_name()
            if run_name:
                handler_kwargs["conversation_id"] = run_name
            return wrapped_query(client=client, _query_fn=call, **handler_kwargs, **kwargs)

        return instrumented

    return wrap


def register_all(client: typing.Any) -> tuple[str, ...]:
    """Register an instrumentor for every framework whose integration is installed.

    Returns the frameworks that were registered, which is what :func:`init` reports back so
    a caller can tell instrumentation is actually in place.
    """
    from flyteplugins.agents.core import register_call_wrapper, register_instrumentor

    registered: list[str] = []
    for integration in SUPPORTED_FRAMEWORKS:
        injector = _load(integration)
        if injector is None:
            continue

        if integration.call_wrapper:
            # agento11y_query instruments the options itself, so registering a payload
            # instrumentor as well would attach the hooks twice.
            register_call_wrapper(integration.framework, _make_call_wrapper(injector, client))
            registered.append(integration.framework)
            continue

        def make(injector: typing.Any = injector, integration: _Integration = integration) -> typing.Any:
            # Bound as defaults so each closure keeps its own integration rather than the
            # last one the loop saw.
            def instrument(payload: typing.Any) -> typing.Any:
                if integration.payload_is_run_kwargs:
                    return _instrument_run_kwargs(payload, injector, integration, client)
                if integration.conversation_via_metadata:
                    payload = _with_conversation_metadata(payload)
                if integration.conversation_via_context:
                    payload = _with_conversation_context(payload)
                kwargs: dict[str, typing.Any] = {"client": client}
                if integration.async_handler:
                    # Every adapter drives its framework asynchronously.
                    kwargs["async_handler"] = True
                return injector(payload, **kwargs)

            return instrument

        register_instrumentor(integration.framework, make())
        registered.append(integration.framework)

    return tuple(registered)


def unregister_all() -> None:
    """Remove every instrumentor and call wrapper this module registered."""
    from flyteplugins.agents.core import unregister_call_wrapper, unregister_instrumentor

    for integration in SUPPORTED_FRAMEWORKS:
        unregister_instrumentor(integration.framework)
        unregister_call_wrapper(integration.framework)
