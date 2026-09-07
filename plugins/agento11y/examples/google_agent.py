"""A Google ADK agent, instrumented for Grafana Agent Observability.

The agent code is unchanged from the plain flyteplugins-agents-google version. Calling init at
module scope is the whole integration: the adapter offers its Runner kwargs to an instrumentor on the
way past, which is how an agento11y ADK plugin reaches a call the adapter owns rather than you.

    pip install "flyteplugins-agento11y[google]"
    python google_agent.py            # on the cluster in your flyte config
    python google_agent.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte
from flyteplugins.agents.google import run_agent, tool
from flyteplugins.otel.grafana import GrafanaTrace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.agento11y import GrafanaAgentObservability, init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so the spans are visible in the task logs."""
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


# Module scope, not inside the task: the task span opens before the task body runs, and the
# Flyte identity binding rides on that span.
init(service_name="google-support", exporter=_console_or_otlp(), disable_batch=True)

image = (
    flyte.Image.from_debian_base()
    # The agento11y framework integration is a PyPI package, not one of ours.
    .with_pip_packages("agento11y-google-adk")
    # Bake the locally-built plugin wheels from ./dist so a cluster run exercises the working
    # tree rather than PyPI releases. Build them with `make dist-all`.
    .with_local_v2_plugins(
        [
            "flyteplugins-agents-core",
            "flyteplugins-agents-google",
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
    name="google_support",
    image=image,
    # agento11y defaults its auth mode to "none", so a token alone is never sent and the
    # export comes back 401. Grafana Cloud uses Basic with the instance id as the username.
    env_vars={"AGENTO11Y_AUTH_MODE": "basic"},
    secrets=[
        flyte.Secret(key="google_api_key", as_env_var="GOOGLE_API_KEY"),
        flyte.Secret(key="agento11y_endpoint", as_env_var="AGENTO11Y_ENDPOINT"),
        flyte.Secret(key="agento11y_token", as_env_var="AGENTO11Y_AUTH_TOKEN"),
        flyte.Secret(key="agento11y_tenant_id", as_env_var="AGENTO11Y_AUTH_TENANT_ID"),
        # Spans go to Tempo over OTLP; generations go to Agent Observability over their own
        # channel. Both are needed for the two links on the task to resolve.
        flyte.Secret(key="otlp_endpoint", as_env_var="OTEL_EXPORTER_OTLP_ENDPOINT"),
        flyte.Secret(key="otlp_headers", as_env_var="OTEL_EXPORTER_OTLP_HEADERS"),
    ],
)


@env.task
async def lookup_order(order_id: str) -> str:
    """A durable Flyte child action, and a tool call in Grafana."""
    return f"Order {order_id} shipped on 2026-07-20."


@env.task(report=True, retries=3, links=links)
async def google_agent(question: str = "Where is order A-1001?") -> str:
    return await run_agent(
        question,
        tools=[tool(lookup_order)],
        model="gemini-3.6-flash",
        instructions="You are a support agent. Use the tools to answer.",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same task
    # in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(google_agent, question="Where is order A-1001?").url)
