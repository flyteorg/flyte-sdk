# flyteplugins-agento11y

Grafana Agent Observability for Flyte agents.

One call sends generations, tool calls, token usage, and cost to Grafana, nested inside the
Flyte task span and grouped by Flyte run.

```python
import flyte
from flyteplugins.agento11y import init
from flyteplugins.agents.openai import run_agent

# Module scope, not inside a task.
init(service_name="my-agent")

env = flyte.TaskEnvironment(name="agent_env")

@env.task
async def agent(question: str) -> str:
    return await run_agent(question, tools=[...], model="gpt-4.1")
```

The agent code is unchanged. Install the extra for your framework, which is what makes its
instrumentor available:

```bash
pip install "flyteplugins-agento11y[openai]"
```

Extras: `langchain`, `langgraph`, `openai`, `claude`, `google`, `pydantic-ai`.

## Why this needs a plugin at all

agento11y's framework integrations attach at the call site. A LangChain callback handler, an
OpenAI Agents `RunHooks`, a `ClaudeAgentOptions` — each is passed into the invocation:

```python
handler = Agento11yLangChainHandler(client=client)
chain.invoke({"input": "Hello"}, config={"callbacks": [handler]})
```

In the Flyte agents plugin the adapter owns that invocation. You call `run_agent(...)`, and
`ainvoke` happens inside the adapter, so there is nowhere to put the handler.

So `flyteplugins-agents-core` offers each framework's native run payload to a registry on the
way past, and this package answers. The agents plugin never imports agento11y; it moves an
opaque object through a function.

## What you get beyond agento11y on its own

agento11y works in a Flyte task without any of this, and Grafana's dashboards will light up,
because they are driven by generation records rather than by trace structure. What is missing
is everything Flyte knows and agento11y cannot.

Without the plugin, three model calls in a task become three unrelated root traces. There is
no task boundary, nothing tying a generation to the run that produced it, and on a resume the
replayed steps produce nothing at all.

With it:

Generations nest inside the Flyte task span, so one run is one trace, and generation records
carry that run's trace id — which is the link from a generation in Grafana back to the Flyte
run.

Flyte identity is bound onto agento11y's own context, so nothing has to be restated:

| agento11y       | Flyte        |
| --------------- | ------------ |
| conversation id | run name     |
| agent name      | task name    |
| agent version   | task version |

A Flyte run therefore shows up in Grafana as one conversation, and a redeploy as a new agent
version, so the before and after of a prompt change is directly comparable.

Durability is preserved end to end. A crashed and resumed run stays a single trace, because
the trace id is derived from the run identity rather than generated per process. Steps the
resumed run replayed from its durable log appear marked `flyte.replayed`, so the trace has no
holes where durability did its job. And those steps do not call the model again, so a resume
does not pay for the generations the first attempt already bought.

## Examples

| Example                 | What it covers                                          |
| ----------------------- | ------------------------------------------------------- |
| `openai_agent.py`       | An OpenAI Agents agent with durable Flyte tools         |
| `langgraph_agent.py`    | LangGraph, which also records workflow steps            |
| `manual_generations.py` | Recording generations by hand, no framework integration |
| `durable_agent.py`      | A crash and its resume, as one trace                    |

## Linking back from Grafana

`GrafanaAgentObservability` links a Flyte action to its conversation in Agent Observability,
rendered in the UI. It works because this package binds the run name as the conversation id.

```python
from flyteplugins.agento11y import GrafanaAgentObservability
from flyteplugins.otel.grafana import GrafanaTrace

@env.task(links=(
    GrafanaAgentObservability(host="https://myorg.grafana.net"),
    GrafanaTrace(host="https://myorg.grafana.net", datasource_uid="<tempo-uid>"),
))
async def agent(question: str) -> str:
    ...
```

The two answer different questions: the first goes to the generations, prompts, and cost;
the second to the distributed trace in Tempo. `GrafanaTrace` lives in `flyteplugins-otel`
because it needs nothing from this package.

The link opens the conversation itself rather than the filtered list, and fills the app's
back navigation with the list scoped to the same run.

`app_id` defaults to `grafana-agento11y-app`. The app was previously `grafana-sigil-app`,
which still resolves but is deprecated, so the id and both path templates are settable in
case it moves again.

## Configuration

With no arguments, `init` reads the standard `AGENTO11Y_*` variables, which is how the
Grafana docs configure it. Supply them as a `flyte.Secret` rather than hardcoding them.

```python
init(
    service_name="my-agent",
    endpoint="https://<your-stack>.grafana.net",   # or AGENTO11Y_ENDPOINT
)
```

Useful arguments:

- `client` — use an agento11y client you built yourself; it is left alone and not shut down.
- `client_options` — extra `ClientConfig` fields: auth mode, protocol, content capture.
- `bind_conversation=False` — keep your own conversation ids, for conversations that span
  more than one run.
- `trace=False` — skip initializing `flyteplugins-otel`, if you configure tracing yourself.
- Anything else is forwarded to `flyteplugins.otel.init`, including `tracer_provider` for an
  OpenTelemetry setup you already have.

## Content capture

agento11y sends metadata by default — model, token usage, tool names, timing — and keeps
prompts and responses local unless you opt in. That is an agento11y setting rather than a
Flyte one; pass it through `client_options`.

## Limitations

crewai and mistral have Flyte adapters but no agento11y integration, so their runs are traced
and their tasks and tool calls appear, but generations are not captured automatically. Record
them by hand as in `manual_generations.py`.

`init` must be called at module scope. The task span opens before the task body runs, so
initializing from inside the body means that task's span, and the identity binding that rides
on it, have already been missed. `flyteplugins-otel` warns when it detects this.
