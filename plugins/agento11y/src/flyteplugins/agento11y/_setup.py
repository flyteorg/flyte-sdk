"""Standing the whole thing up in one call."""

from __future__ import annotations

import atexit
import os
import threading
import typing

import flyte._observe as observe
from flyte._logging import logger

from agento11y import Client, ClientConfig

from ._binding import FlyteIdentityBinding
from ._frameworks import register_all, unregister_all

__all__ = ["get_client", "init", "instrumented_frameworks", "shutdown"]

# Process-global for the same reason as flyteplugins-otel: the identity binding is registered
# on flyte._observe's module-level observer list, so this must not be context-local. The lock
# makes the check-then-set in init atomic.
_lock = threading.Lock()
_state: dict[str, typing.Any] = {"client": None, "binding": None, "owns_client": False, "frameworks": ()}


def init(
    *,
    service_name: str | None = None,
    endpoint: str | None = None,
    client: Client | None = None,
    client_options: typing.Mapping[str, typing.Any] | None = None,
    bind_conversation: bool = True,
    bind_agent_name: bool = True,
    trace: bool = True,
    **otel_kwargs: typing.Any,
) -> Client:
    """Send Flyte agent runs to Grafana Agent Observability.

    Call this at module scope, not inside a task. Task spans open before the task body runs,
    so initializing from within the body means that task's span, and the identity binding
    that rides on it, have already been missed.

    Three things get wired up. An agento11y client, sharing the tracer from
    `flyteplugins-otel` so generation spans nest inside the Flyte task span rather than
    floating off as their own traces. A binding that hands Flyte's run, task, and version to
    agento11y as conversation id, agent name, and agent version. And an instrumentor for each
    agent framework whose integration package is installed, which is what lets the agents
    plugin attach handlers to calls it owns.

    Args:
        service_name: Value for service.name on the OpenTelemetry side.
        endpoint: agento11y generation export endpoint. Falls back to the AGENTO11Y_ env
            vars, which is how the Grafana docs configure it.
        client: Use a client you built yourself. Its tracer and exporters are left alone,
            and it is not shut down by `shutdown`.
        client_options: Extra `ClientConfig` fields, for anything this signature does not
            surface: auth mode and token, protocol, content capture, a custom
            `generation_exporter`. Ignored when `client` is supplied.
        bind_conversation: Bind the Flyte run name as agento11y's conversation id. Turn this
            off if conversations in your product span more than one run.
        bind_agent_name: Bind the Flyte task name as agento11y's agent name. Turn this off for
            a task that drives more than one agent, so each keeps its framework-given name
            instead of all of them reporting as the task.
        trace: Also initialize `flyteplugins-otel`, so Flyte tasks and trace steps become
            spans and generations nest inside them. Turn it off if you initialize it
            yourself, or if you only want generations.
        **otel_kwargs: Passed through to `flyteplugins.otel.init` (`endpoint`,
            `headers`, `tracer_provider`, `disable_batch`, and so on).

    Returns:
        The agento11y client, for creating generations directly.
    """
    # Held across the whole of init so the check and the registration cannot interleave.
    # flyteplugins-otel takes its own separate lock inside otel_init, so there is no cycle.
    with _lock:
        return _init_locked(
            service_name=service_name,
            endpoint=endpoint,
            client=client,
            client_options=client_options,
            bind_conversation=bind_conversation,
            bind_agent_name=bind_agent_name,
            trace=trace,
            otel_kwargs=otel_kwargs,
        )


def _init_locked(
    *,
    service_name: str | None,
    endpoint: str | None,
    client: Client | None,
    client_options: typing.Mapping[str, typing.Any] | None,
    bind_conversation: bool,
    bind_agent_name: bool,
    trace: bool,
    otel_kwargs: dict[str, typing.Any],
) -> Client:
    """The body of `init`, run with the lock already held."""
    if _state["client"] is not None:
        logger.debug("flyteplugins-agento11y already initialized; returning the existing client")
        return _state["client"]

    tracer = None
    if trace:
        from flyteplugins.otel import get_tracer
        from flyteplugins.otel import init as otel_init

        otel_init(service_name=service_name, **otel_kwargs)
        tracer = get_tracer()

    owns_client = client is None
    if client is None:
        config_kwargs: dict[str, typing.Any] = {}
        if tracer is not None:
            config_kwargs["tracer"] = tracer
        if endpoint:
            config_kwargs["generation_export_endpoint"] = endpoint
        elif not os.environ.get("AGENTO11Y_ENDPOINT"):
            # With nowhere to send them, agento11y falls back to gRPC on localhost:4317 and
            # every flush produces connection errors. That is the ordinary state of the
            # process that submits a run, and of any example run without credentials, so say
            # it once and drop the generations instead.
            from agento11y.exporters.noop import NoopGenerationExporter

            config_kwargs["generation_exporter"] = NoopGenerationExporter()
            logger.warning(
                "flyteplugins-agento11y: no AGENTO11Y_ENDPOINT configured, so generations are recorded "
                "but not exported. Set the variable or pass endpoint= to send them to Grafana. "
                "OpenTelemetry spans are unaffected and still follow their own exporter settings."
            )
        # Caller-supplied options win, so anything this signature does not cover stays
        # reachable — and the exporter in particular has to be set before construction.
        config_kwargs.update(client_options or {})
        client = Client(ClientConfig(**config_kwargs))

    if owns_client:
        # agento11y batches generations and flushes on an interval, and unlike OpenTelemetry's
        # TracerProvider it registers no exit hook of its own. A task that finishes inside the
        # flush window therefore exits with its generations still buffered, and they are lost:
        # the agent never appears in Grafana at all. Short tasks are exactly the ones that hit
        # this, so the flush cannot be left to the user remembering to call shutdown.
        atexit.register(_flush_at_exit)

    binding = FlyteIdentityBinding(bind_conversation=bind_conversation, bind_agent_name=bind_agent_name)
    observe.register_observer(binding)
    frameworks = register_all(client)

    _state.update({"client": client, "binding": binding, "owns_client": owns_client, "frameworks": frameworks})
    logger.debug(f"flyteplugins-agento11y instrumenting: {', '.join(frameworks) or 'no frameworks installed'}")
    return client


def _flush_at_exit() -> None:
    """Shut the client down on interpreter exit, so buffered generations are not dropped.

    Deliberately quiet and best effort: this runs while the process is already on its way
    out, and a failure here must not turn a successful task into a crash on the way down.
    """
    try:
        shutdown()
    except Exception:  # pragma: no cover - interpreter teardown
        logger.debug("agento11y flush at exit failed", exc_info=True)


def get_client() -> Client | None:
    """The client `init` built, for recording generations by hand."""
    return _state["client"]


def instrumented_frameworks() -> tuple[str, ...]:
    """Frameworks that got an instrumentor, that is, whose integration package is installed."""
    return _state["frameworks"]


def shutdown() -> None:
    """Unbind, unregister, and flush a client we created."""
    with _lock:
        _shutdown_locked()


def _shutdown_locked() -> None:
    binding = _state["binding"]
    if binding is not None:
        observe.unregister_observer(binding)
    unregister_all()

    # A client handed to us belongs to the caller and may still be in use.
    if _state["owns_client"] and _state["client"] is not None:
        with_suppressed = getattr(_state["client"], "shutdown", None)
        if with_suppressed is not None:
            try:
                with_suppressed()
            except Exception:
                logger.debug("agento11y client shutdown failed", exc_info=True)

    _state.update({"client": None, "binding": None, "owns_client": False, "frameworks": ()})
