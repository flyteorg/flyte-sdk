"""An agent in a sync task, instrumented the same as an async one.

Every adapter has a ``run_agent_sync`` for tasks written with ``def`` rather than ``async
def``. It runs the async implementation on a background event loop, so the call crosses a
thread boundary — and the instrumentation rides on contextvars, which is what thread
boundaries tend to drop.

It survives the crossing because the bridge dispatches through
``asyncio.run_coroutine_threadsafe``, which copies the calling thread's context before
handing the coroutine over. The span the observer made current, the agento11y conversation
binding and the Flyte task context are all opened around the task body, so they are in that
copy by the time the agent runs. A sync task therefore produces the same task span, the same
nested generations, and the same conversation keyed by Flyte run as its async equivalent.

Propagation is one-way: values set inside the agent do not come back out. Nothing here needs
them to. Against openai_agent.py this file changes only ``def`` for ``async def`` and
``run_agent_sync`` for ``await run_agent`` — nothing in the setup above the tasks differs.

The tools are sync here too. A tool is a Flyte task either way, so it still runs as a durable
child action in its own container rather than inline in the agent's process.

    pip install "flyteplugins-agento11y[openai]"
    python sync_agent.py            # on the cluster in your flyte config
    python sync_agent.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte
from flyteplugins.agents.openai import run_agent_sync, tool
from flyteplugins.otel.grafana import GrafanaTrace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.agento11y import GrafanaAgentObservability, init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so the spans are visible in the task logs."""
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


# Module scope, not inside the task: the task span opens before the task body runs, and the
# Flyte identity binding rides on that span.
init(service_name="sync-support", exporter=_console_or_otlp(), disable_batch=True)

image = (
    flyte.Image.from_debian_base()
    # The agento11y framework integration is a PyPI package, not one of ours.
    .with_pip_packages("agento11y-openai-agents")
    # Bake the locally-built plugin wheels from ./dist so a cluster run exercises the working
    # tree rather than PyPI releases. Build them with `make dist-all`.
    .with_local_v2_plugins(
        [
            "flyteplugins-agents-core",
            "flyteplugins-agents-openai",
            "flyteplugins-otel",
            "flyteplugins-agento11y",
        ]
    )
)

# Links rendered on the action in the Flyte UI: the run's conversation in Agent Observability,
# and its spans in Tempo. Skipped when GRAFANA_HOST is unset.
GRAFANA_HOST = os.environ.get("GRAFANA_HOST", "")
TEMPO_DATASOURCE_UID = os.environ.get("TEMPO_DATASOURCE_UID", "grafanacloud-traces")

links = (
    (
        GrafanaAgentObservability(host=GRAFANA_HOST),
        GrafanaTrace(host=GRAFANA_HOST, datasource_uid=TEMPO_DATASOURCE_UID),
    )
    if GRAFANA_HOST
    else ()
)

env = flyte.TaskEnvironment(
    name="sync_support",
    image=image,
    # agento11y defaults its auth mode to "none", so a token alone is never sent and the
    # export comes back 401. Grafana Cloud uses Basic with the instance id as the username.
    env_vars={"AGENTO11Y_AUTH_MODE": "basic"},
    secrets=[
        flyte.Secret(key="sam_openai_api_key", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="sam_agento11y_endpoint", as_env_var="AGENTO11Y_ENDPOINT"),
        flyte.Secret(key="sam_agento11y_token", as_env_var="AGENTO11Y_AUTH_TOKEN"),
        flyte.Secret(key="sam_agento11y_tenant_id", as_env_var="AGENTO11Y_AUTH_TENANT_ID"),
        # Spans go to Tempo over OTLP; generations go to Agent Observability over their own
        # channel. Both are needed for the two links on the task to resolve.
        flyte.Secret(key="sam_otlp_endpoint", as_env_var="OTEL_EXPORTER_OTLP_ENDPOINT"),
        flyte.Secret(key="sam_otlp_headers", as_env_var="OTEL_EXPORTER_OTLP_HEADERS"),
    ],
)


@env.task
def lookup_order(order_id: str) -> str:
    """A sync tool, and a durable Flyte child action in its own container."""
    return f"Order {order_id} shipped on 2026-07-20."


# `def`, not `async def`. run_agent_sync drives the agent from here.
@env.task(report=True, retries=3, links=links)
def sync_agent(question: str = "Where is order A-1001?") -> str:
    return run_agent_sync(
        question,
        tools=[tool(lookup_order)],
        model="gpt-4.1",
        instructions="You are a support agent. Use the tools to answer.",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same task
    # in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(sync_agent, question="Where is order A-1001?").url)
