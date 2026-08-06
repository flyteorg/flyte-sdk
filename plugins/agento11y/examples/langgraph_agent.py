"""A LangGraph agent, instrumented for Grafana Agent Observability.

Same shape as every other framework: init at module scope, agent code untouched. For
LangGraph the instrumentor attaches a callback handler to the runnable config the adapter
passes to ainvoke.

LangGraph is one of the integrations that records workflow steps as well as generations, so
non-LLM nodes (routing, retrieval) show up in Grafana too, not just model calls.

    pip install "flyteplugins-agento11y[langgraph]"
    export AGENTO11Y_ENDPOINT=... AGENTO11Y_AUTH_TOKEN=...
    python langgraph_agent.py            # on the cluster in your flyte config
    python langgraph_agent.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte
from flyteplugins.agents.langgraph import run_agent, tool
from flyteplugins.otel.grafana import GrafanaTrace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.agento11y import GrafanaAgentObservability, init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so the spans are visible in the task logs.

    Generations go to Grafana over their own channel regardless; this only decides where the
    spans land.
    """
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


init(service_name="research-agent", exporter=_console_or_otlp(), disable_batch=True)

image = (
    flyte.Image.from_debian_base()
    .with_pip_packages("agento11y-langgraph", "langchain-openai")
    # Bake the locally-built plugin wheels from ./dist so a remote run exercises the working
    # tree rather than PyPI releases. Build them with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(
        [
            "flyteplugins-agents-core",
            "flyteplugins-agents-langgraph",
            "flyteplugins-otel",
            "flyteplugins-agento11y",
        ]
    )
)

# Links rendered on the action in the Flyte UI. The first goes to this run's conversation in
# Agent Observability, which works because the bridge binds the run name as the conversation
# id; the second to its spans in Tempo. Both need your stack URL, and the trace link also
# needs the Tempo datasource UID, so they are skipped when GRAFANA_HOST is unset.
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
    name="research",
    image=image,
    # agento11y defaults its auth mode to "none", so a token alone is never sent and the
    # export comes back 401. Grafana Cloud uses Basic with the instance id as the username,
    # which agento11y fills from AGENTO11Y_AUTH_TENANT_ID when the mode is "basic".
    env_vars={"AGENTO11Y_AUTH_MODE": "basic"},
    secrets=[
        flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="agento11y_endpoint", as_env_var="AGENTO11Y_ENDPOINT"),
        flyte.Secret(key="agento11y_token", as_env_var="AGENTO11Y_AUTH_TOKEN"),
        flyte.Secret(key="agento11y_tenant_id", as_env_var="AGENTO11Y_AUTH_TENANT_ID"),
        # Spans go to Tempo over OTLP; generations go to Agent Observability over their
        # own channel. Both are needed for the two links on the task to resolve.
        flyte.Secret(key="otlp_endpoint", as_env_var="OTEL_EXPORTER_OTLP_ENDPOINT"),
        flyte.Secret(key="otlp_headers", as_env_var="OTEL_EXPORTER_OTLP_HEADERS"),
    ],
)


@env.task
async def search(query: str) -> str:
    return f"Results for {query}: Flyte is a workflow orchestrator."


@env.task(report=True, retries=3, links=links)
async def research_agent(question: str = "What is Flyte?") -> str:
    from langchain_openai import ChatOpenAI

    return await run_agent(
        question,
        tools=[tool(search)],
        model=ChatOpenAI(model="gpt-4o"),
        instructions="Answer using the search tool.",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(research_agent, question="What is Flyte?").url)
