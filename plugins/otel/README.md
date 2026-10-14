# flyteplugins-otel

OpenTelemetry tracing for Flyte.

Every task becomes a span, every `flyte.trace` step becomes a child span inside it, and spans
created by your own code or by any instrumentation library nest underneath without wiring.
Export goes wherever OTLP goes.

None of this is specific to agents or to LLM work. It is ordinary distributed tracing for
ordinary Flyte tasks with some additions for the things Flyte does that a normal
OpenTelemetry setup has no way to model.

## Install and use

```python
import flyte
from flyteplugins.otel import init

# At module scope, not inside a task. The task span opens before the task body runs, so
# initializing from within the body means that task's own span has already been missed.
init(service_name="my-service")

env = flyte.TaskEnvironment(name="my_env")

@env.task
async def main(n: int) -> int:
    ...
```

With no arguments `init` reads the standard `OTEL_EXPORTER_OTLP_ENDPOINT` and
`OTEL_EXPORTER_OTLP_HEADERS` variables, which is the shape most vendors document. Supply them
as a `flyte.Secret` rather than hardcoding them.

A base gateway URL is fine if you pass the endpoint directly; the traces path is appended:

```python
init(
    service_name="my-service",
    endpoint="https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
    headers={"Authorization": "Basic <base64>"},
)
```

### Exporters

OTLP is the default but not a requirement. Any `SpanExporter` works and several can run side by
side — a console exporter alongside a real backend, say:

```python
init(exporter=[ConsoleSpanExporter(), JaegerExporter(...)])
```

For OTLP, both transports are supported. The protocol follows
`OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`, then `OTEL_EXPORTER_OTLP_PROTOCOL`, then
`http/protobuf`, or pass it directly. gRPC ships separately:

```bash
pip install "flyteplugins-otel[grpc]"
```

```python
init(endpoint="http://collector:4317", protocol="grpc")
```

If you already configure OpenTelemetry yourself, hand over the provider instead and nothing
about your setup changes:

```python
init(tracer_provider=my_provider)
```

## Examples

| Example                    | What it covers                                          |
| -------------------------- | ------------------------------------------------------- |
| `basic_tracing.py`         | The smallest setup, printing spans to the console       |
| `custom_spans.py`          | Your own spans inside a task, nesting automatically     |
| `nested_tasks.py`          | Tasks calling tasks, across pods, in one trace          |
| `existing_provider.py`     | Adopting a TracerProvider you already configured        |
| `propagate_from_caller.py` | Joining a trace that started outside Flyte              |
| `http_instrumentation.py`  | Third party auto-instrumentation, here httpx            |
| `grafana_cloud.py`         | Exporting to a real backend, with secrets               |
| `durable_trace.py`         | A crash and its resume, as a single trace               |

## Trace context, in and out

Flyte propagates a key-value `custom_context` through a run and into every sub-action. This
plugin uses it as a W3C carrier in both directions.

Inbound: if `custom_context` holds a `traceparent`, the task span starts under it rather than
becoming a root. So a run submitted from inside a caller's span joins the caller's trace:

```python
with tracer.start_as_current_span("incoming_request"):
    carrier = {}
    inject(carrier)
    run = flyte.with_runcontext(custom_context=carrier).run(main, ...)
```

Outbound: once the task span is open, the plugin publishes it back into `custom_context`, so
a child task running in another pod nests under the task that spawned it with nothing passed
by hand. Because `custom_context` travels in the action's persisted inputs, this survives a
resume too.

Which means the manual `extract` inside each task is no longer needed just to get spans
parented correctly. Reach for it when you want to start your own spans under the incoming
context; the plugin's own spans are already there.

## What durability adds

A run that crashes and resumes is several processes over time. Each starts a fresh
OpenTelemetry SDK that has no idea the earlier ones existed, and two things go wrong.

Every attempt mints its own random trace id, so one agent run arrives at the backend as
several unrelated traces. And every step the resumed run replayed out of its durable log is
missing entirely, because replayed steps never execute, so nothing instruments them. The
trace ends up with holes in it exactly where durability did its job.

So when there is no inbound trace context, the trace id is derived from the run identity
rather than generated. Every process computes the same value from information it already has,
and all of them record into one trace with no coordination. Span ids stay random, so each
attempt is a distinct subtree: a resumed run reads as the attempt that crashed followed by
the attempt that finished.

Replayed steps are recorded as spans marked `flyte.replayed`. They have no meaningful
duration, because no work happened, but they are present, so the trace is complete.

For a live demo, `disable_batch=True` exports each span as it ends. It is slower, but nothing
is lost when the process dies, which matters when the thing being demonstrated is a crash.

## Linking back from Grafana

`flyteplugins.otel.grafana` builds links from a Flyte action into Grafana, rendered in the
Flyte UI. They are plain URL builders with no Grafana dependency.

```python
from flyteplugins.otel.grafana import GrafanaTrace

@env.task(links=(GrafanaTrace(host="https://myorg.grafana.net", datasource_uid="<tempo-uid>"),))
async def my_task() -> str:
    ...
```

The trace link queries on `flyte.run_name` rather than addressing a trace by id, so it finds
a run's spans whatever their trace ids turn out to be — including runs whose trace context
came from outside. It embeds a time range because Explore otherwise defaults to the last
hour, and a link to an older run would open on an empty pane.

The datasource UID is per-stack and not guessable; find it under Connections, Data sources.

Flyte hands links placeholders such as `{{.runName}}` when the task is serialized and swaps
them for real values on the finished URI, so the link is built once and works for every run.
That substitution is a plain string replacement, which means the placeholders have to survive
URL encoding intact — the link builders keep them literal for exactly this reason.

Only the Tempo link lives here. A link to a run's conversation in Grafana Agent
Observability ships with `flyteplugins-agento11y`, since it is that package's identity
binding that makes a run addressable by conversation id at all.

## Span attributes

Every span carries the identifiers needed to get back to the run that produced it. These are
effectively public API, since a Grafana data link queries on them to jump from a span into
the Flyte UI.

| Attribute                                    | On         | Meaning                                           |
| -------------------------------------------- | ---------- | ------------------------------------------------- |
| `flyte.run_name`                             | all        | The run, and what the trace id is derived from    |
| `flyte.action_name`                          | all        | The action that produced the span                 |
| `flyte.project`, `flyte.domain`, `flyte.org` | all        | Where the run lives                               |
| `flyte.task_name`                            | task spans | The task being executed                           |
| `flyte.step_name`                            | step spans | The traced function                               |
| `flyte.task_action_name`                     | step spans | The task that owns the step                       |
| `flyte.replayed`                             | step spans | Whether this step was served from the durable log |

## Using it alongside other instrumentation

Libraries that emit their own spans — the OpenTelemetry instrumentation packages, agent
observability SDKs, vendor SDKs — need no extra wiring. Parenting comes from the active
context rather than from the tracer provider, and the task span is active for the whole task
body, so their spans land inside it.

Calling `init` before the other library keeps everything on one export pipeline.
`get_tracer()` returns the tracer this plugin built for libraries that accept one.

## Flyte's own control-plane calls appear as spans

Once tracing is on you will see `POST` client spans for Flyte's calls to the control plane —
`Enqueue`, `CreateRun`, `UploadInputs` and so on — alongside your own.

They do not come from this plugin. Flyte's transport is `pyqwest`, whose `HTTPTransport`
takes `enable_otel: bool = True` and falls back to the global tracer provider when it is not
given one. `init(set_global=True)`, the default, installs that provider, so the transport
starts recording through it.

Mostly this is useful: inside a task the spans nest under the task span, so you can see how
much of a task's wall clock went on talking to the control plane, and the 401-then-200 pairs
show the auth retry. Two things to be aware of. The volume scales with sub-action count, so a
wide fan-out produces a lot of them. And calls made outside a task span, during submission,
arrive as their own root traces rather than joining the run's trace.

There is no switch for this in the plugin, since the transport is Flyte's rather than ours.
`init(set_global=False)` keeps the provider out of the global slot, which stops the transport
finding it, at the cost of other instrumentation not finding it either.

## Limitations

With no OTLP endpoint configured — no `OTEL_EXPORTER_OTLP_ENDPOINT`, no `endpoint=`, no
explicit exporter — spans are recorded but not exported, and `init` says so once. This is the
normal state of the process that submits a run, since it imports your module and therefore
runs `init` without needing to export anything. If you run a local collector, point the
variable at it explicitly rather than relying on the OTLP default of `localhost:4318`.

Replayed spans have no duration. The original timing is written to the control plane but does
not come back over the channel a resumed run reads from, so recovering it needs a backend
change. What you get is the step's presence, identity, and outcome.

Trace context rides in `custom_context`, which is a flat string map that Flyte propagates
wholesale. The `traceparent` key is therefore visible to task code and will be overwritten if
something else writes that key.
