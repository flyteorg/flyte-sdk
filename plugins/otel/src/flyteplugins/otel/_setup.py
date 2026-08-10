"""Wiring the observer up to an exporter.

OTLP is the default because it is what most backends document but nothing here requires it.
Any `SpanExporter` works, several can run side by side, and a provider you configured
yourself is adopted whole.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any, Mapping, Optional, Union

import flyte._observe as observe
from flyte._logging import logger
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

from ._observer import OtelObserver, adopt_provider, make_id_generator

__all__ = ["get_tracer", "init", "shutdown"]

_INSTRUMENTATION_NAME = "flyteplugins-otel"

# Process-global, not context-local, and deliberately so: init installs an OpenTelemetry
# TracerProvider (a process global itself) and appends to flyte._observe's module-level
# observer list. Context-local state here would let shutdown miss an observer that is still
# firing for every task. The lock makes the check-then-set in init atomic, matching how
# flyte's own _ControllerState guards the same shape.
_lock = threading.Lock()
_state: dict[str, Any] = {"provider": None, "observer": None, "tracer": None}


def _normalize_headers(headers: Union[Mapping[str, str], str, None]) -> Optional[dict[str, str]]:
    """Accept either a mapping or the comma separated form used by the OTEL_ env vars."""
    if headers is None or isinstance(headers, dict):
        return dict(headers) if headers else None
    if isinstance(headers, Mapping):
        return dict(headers)
    parsed: dict[str, str] = {}
    for pair in str(headers).split(","):
        if not pair.strip():
            continue
        key, _, value = pair.partition("=")
        parsed[key.strip()] = value.strip()
    return parsed or None


def _traces_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """Point an OTLP gateway base URL at its traces path.

    Backends hand out a base URL (Grafana Cloud gives you the OTLP gateway), while the HTTP
    exporter wants the signal specific path when the endpoint is passed explicitly. Doing
    the append here means the value copied out of a vendor console works as-is.
    """
    if not endpoint:
        return None
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/v1/traces"):
        return trimmed
    return f"{trimmed}/v1/traces"


def _resolve_protocol(protocol: Optional[str]) -> str:
    """Pick the OTLP wire protocol, honouring OpenTelemetry's own environment variable.

    The spec spells these "http/protobuf" and "grpc"; the short forms are accepted because
    that is what people type. Ignoring OTEL_EXPORTER_OTLP_PROTOCOL would mean a backend
    configured entirely through the standard variables silently got the wrong transport.
    """
    raw = (
        protocol
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    )
    value = (raw or "http/protobuf").strip().lower()
    if value == "grpc":
        return "grpc"
    if value in ("http", "http/protobuf", "httpprotobuf", "http/proto"):
        return "http/protobuf"
    raise ValueError(f"Unsupported OTLP protocol {raw!r}. Use 'http/protobuf' or 'grpc'.")


def _otlp_exporter(
    *,
    protocol: str,
    endpoint: Optional[str],
    headers: Union[Mapping[str, str], str, None],
) -> SpanExporter:
    """Build the OTLP exporter for the chosen transport.

    gRPC lives in a separate distribution, so its absence is reported as something to install
    rather than an ImportError from deep inside the SDK. Note the endpoint differs by
    transport: gRPC takes the base endpoint, HTTP wants the signal specific path.
    """
    normalized_headers = _normalize_headers(headers)
    if protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
        except ImportError as e:
            raise ImportError(
                "OTLP over gRPC needs the opentelemetry-exporter-otlp-proto-grpc package. "
                'Install it with `pip install "flyteplugins-otel[grpc]"`, or use protocol="http/protobuf".'
            ) from e
        return GrpcExporter(endpoint=endpoint or None, headers=normalized_headers)

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpExporter

    return HttpExporter(endpoint=_traces_endpoint(endpoint), headers=normalized_headers)


def _otlp_endpoint_configured(endpoint: Optional[str]) -> bool:
    """Whether anyone actually said where traces should go.

    Without this the OTLP exporter silently falls back to the spec default of
    http://localhost:4318, and every batch then produces several lines of connection-refused
    retries. That is noise rather than information, and it is the normal state of the driver
    process, which imports the module to submit a run and has no reason to export anything.
    """
    return bool(
        endpoint
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


def _as_exporters(exporter: Union[SpanExporter, Sequence[SpanExporter], None]) -> list[SpanExporter]:
    """Accept one exporter or several. A single SpanExporter is not itself a sequence."""
    if exporter is None:
        return []
    if isinstance(exporter, SpanExporter):
        return [exporter]
    return list(exporter)


def _warn_if_task_already_started() -> None:
    """Say something when init lands too late to record the task's own span.

    The task span opens before the task body runs, so an observer registered from inside the
    body has already missed it. The symptom is a trace holding the trace steps but no task
    span to hang them from, which is hard to work out from the output alone.
    """
    try:
        from flyte._context import internal_ctx

        # Not `flyte.ctx() is not None`: outside a task that returns a falsy NullTaskContext
        # rather than None, so the identity check would report every call as being in a task.
        in_task = internal_ctx().is_task_context()
    except Exception:  # pragma: no cover - context lookup should not break init
        return

    if in_task:
        logger.warning(
            "flyteplugins-otel was initialized from inside a running task, so that task's own span "
            "has already started and will not be recorded. Call init() at module scope instead, so "
            "the observer is registered before any task begins."
        )


def init(
    *,
    service_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    headers: Union[Mapping[str, str], str, None] = None,
    protocol: Optional[str] = None,
    resource_attributes: Optional[Mapping[str, Any]] = None,
    exporter: Union[SpanExporter, Sequence[SpanExporter], None] = None,
    tracer_provider: Optional[trace_api.TracerProvider] = None,
    disable_batch: bool = False,
    set_global: bool = True,
) -> OtelObserver:
    """Start recording Flyte tasks and trace steps as OpenTelemetry spans.

    Call this once at module scope, not inside a task. The task span opens before the task
    body runs, so initializing from within the body means that task's own span has already
    been missed. Module scope runs during import, which happens before any task starts.

    With no arguments it reads the standard OTEL_EXPORTER_OTLP_ENDPOINT and
    OTEL_EXPORTER_OTLP_HEADERS variables, which is the shape most vendors document.

    If you already configure OpenTelemetry yourself, pass `tracer_provider` and none of the
    exporter arguments. The provider is adopted as it stands, with its sampler, resource, and
    exporters untouched; only its id generator is wrapped, so run derived trace ids keep
    working without you giving up your own setup.

    Args:
        service_name: Value for service.name. Defaults to OTEL_SERVICE_NAME, then "flyte".
            Not used when adopting a provider, which carries its own resource.
        endpoint: OTLP endpoint. On http/protobuf a base gateway URL is fine and the traces
            path is added; on grpc the base endpoint is used as given.
        headers: Export headers, as a mapping or the "k=v,k2=v2" form.
        protocol: OTLP transport, "http/protobuf" or "grpc". Defaults to
            OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, then OTEL_EXPORTER_OTLP_PROTOCOL, then
            http/protobuf. gRPC needs the [grpc] extra.
        resource_attributes: Extra resource attributes to attach to every span.
        exporter: Export through this instead of building an OTLP exporter, or pass several
            to fan out — a ConsoleSpanExporter alongside a real backend, say. Any
            `SpanExporter` works; nothing here requires OTLP.
        tracer_provider: Adopt a provider you configured yourself instead of building one.
            Cannot be combined with the exporter building arguments.
        disable_batch: Export each span as it ends. Slower, but nothing is lost if the
            process dies, which matters when the thing being demonstrated is a crash.
        set_global: Install the provider as the global one, so other instrumentation shares
            it. Not used when adopting a provider, which is assumed to be installed already.

    Returns:
        The registered observer, which can be passed to `shutdown`.

    Raises:
        ValueError: If tracer_provider is combined with the exporter building arguments.
    """
    with _lock:
        if _state["observer"] is not None:
            logger.debug("flyteplugins-otel already initialized; returning the existing observer")
            return _state["observer"]

        return _init_locked(
            service_name=service_name,
            endpoint=endpoint,
            headers=headers,
            protocol=protocol,
            resource_attributes=resource_attributes,
            exporter=exporter,
            tracer_provider=tracer_provider,
            disable_batch=disable_batch,
            set_global=set_global,
        )


def _init_locked(
    *,
    service_name: Optional[str],
    endpoint: Optional[str],
    headers: Union[Mapping[str, str], str, None],
    protocol: Optional[str],
    resource_attributes: Optional[Mapping[str, Any]],
    exporter: Union[SpanExporter, Sequence[SpanExporter], None],
    tracer_provider: Optional[trace_api.TracerProvider],
    disable_batch: bool,
    set_global: bool,
) -> OtelObserver:
    """The body of `init` run with the lock already held."""
    _warn_if_task_already_started()

    if tracer_provider is not None:
        conflicting = {
            "endpoint": endpoint,
            "headers": headers,
            "exporter": exporter,
            "protocol": protocol,
            "resource_attributes": resource_attributes,
        }
        supplied = sorted(name for name, value in conflicting.items() if value is not None)
        if supplied:
            raise ValueError(
                f"tracer_provider cannot be combined with {', '.join(supplied)}: the supplied provider "
                f"already owns its exporters and resource. Configure them on the provider instead."
            )
        return _attach(tracer_provider, owns_provider=False)

    attributes: dict[str, Any] = {
        "service.name": service_name or os.environ.get("OTEL_SERVICE_NAME") or "flyte",
    }
    if resource_attributes:
        attributes.update(resource_attributes)

    provider = TracerProvider(resource=Resource.create(attributes), id_generator=make_id_generator())

    exporters = _as_exporters(exporter)
    if not exporters:
        if _otlp_endpoint_configured(endpoint):
            exporters = [_otlp_exporter(protocol=_resolve_protocol(protocol), endpoint=endpoint, headers=headers)]
        else:
            # Say it once, here, rather than let it surface as a retry storm from the
            # exporter. Spans are still recorded, so nesting and context propagation behave
            # exactly as they would; they are simply dropped instead of shipped.
            logger.warning(
                "flyteplugins-otel: no OTLP endpoint configured, so spans are recorded but not exported. "
                "Set OTEL_EXPORTER_OTLP_ENDPOINT, pass endpoint=, or pass an exporter such as "
                "ConsoleSpanExporter. If you are running a local collector, point the variable at it "
                "explicitly (http://localhost:4318) rather than relying on the default."
            )

    # One processor per exporter, so several can run side by side.
    for each in exporters:
        provider.add_span_processor(SimpleSpanProcessor(each) if disable_batch else BatchSpanProcessor(each))

    if set_global:
        trace_api.set_tracer_provider(provider)

    return _attach(provider, owns_provider=True)


def _attach(provider: trace_api.TracerProvider, *, owns_provider: bool) -> OtelObserver:
    """Register an observer against a provider, remembering whether we may shut it down."""
    if not owns_provider and not adopt_provider(provider):
        logger.debug(
            "The supplied tracer provider has no id generator to wrap. Spans are still recorded, but trace "
            "ids will not be derived from the run, so separate attempts of a run will not share a trace."
        )

    tracer = provider.get_tracer(_INSTRUMENTATION_NAME)
    observer = OtelObserver(tracer)
    observe.register_observer(observer)

    # Only a provider we built is ours to shut down; adopting one must not flush or close it.
    _state.update({"provider": provider if owns_provider else None, "observer": observer, "tracer": tracer})
    return observer


def get_tracer() -> Optional[trace_api.Tracer]:
    """The tracer `init` built for handing to another instrumentation library.

    Libraries that create their own provider still nest correctly since parenting comes from
    the active context rather than the provider. Sharing the tracer just keeps everything on
    one export pipeline.
    """
    return _state["tracer"]


def shutdown() -> None:
    """Unregister the observer and flush pending spans."""
    with _lock:
        _shutdown_locked()


def _shutdown_locked() -> None:
    observer = _state["observer"]
    if observer is not None:
        observe.unregister_observer(observer)
    provider = _state["provider"]
    if provider is not None:
        provider.shutdown()
    _state.update({"provider": None, "observer": None, "tracer": None})
