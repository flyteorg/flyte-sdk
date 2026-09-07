"""Recording generations by hand for a framework with no integration.

crewai and mistral have Flyte adapters but no agento11y package yet, and plenty of agents are
written against a provider SDK directly. The client is still there, so generations can be
recorded explicitly, and they still land inside the Flyte task span and carry the run's
identity because that part does not depend on any framework integration.

    pip install flyteplugins-agento11y
    python manual_generations.py            # on the cluster in your flyte config
    python manual_generations.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte
from agento11y import GenerationStart, ModelRef, assistant_text_message, user_text_message
from flyteplugins.otel.grafana import GrafanaTrace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.agento11y import GrafanaAgentObservability, get_client, init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so this runs with or without a backend."""
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


init(service_name="manual-agent", exporter=_console_or_otlp(), disable_batch=True)

image = (
    flyte.Image.from_debian_base()
    # Bake the locally-built plugin wheels from ./dist so a cluster run exercises the working
    # tree rather than PyPI releases. Build them with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(["flyteplugins-agents-core", "flyteplugins-otel", "flyteplugins-agento11y"])
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
    name="manual",
    image=image,
    # agento11y defaults its auth mode to "none", so a token alone is never sent and the
    # export comes back 401. Grafana Cloud uses Basic with the instance id as the username,
    # which agento11y fills from AGENTO11Y_AUTH_TENANT_ID when the mode is "basic".
    env_vars={"AGENTO11Y_AUTH_MODE": "basic"},
    secrets=[
        flyte.Secret(key="agento11y_endpoint", as_env_var="AGENTO11Y_ENDPOINT"),
        flyte.Secret(key="agento11y_token", as_env_var="AGENTO11Y_AUTH_TOKEN"),
        flyte.Secret(key="agento11y_tenant_id", as_env_var="AGENTO11Y_AUTH_TENANT_ID"),
        # Spans go to Tempo over OTLP; generations go to Agent Observability over their
        # own channel. Both are needed for the two links on the task to resolve.
        flyte.Secret(key="otlp_endpoint", as_env_var="OTEL_EXPORTER_OTLP_ENDPOINT"),
        flyte.Secret(key="otlp_headers", as_env_var="OTEL_EXPORTER_OTLP_HEADERS"),
    ],
)


@flyte.trace
async def ask(question: str) -> str:
    """A durable model turn, recorded as a generation."""
    client = get_client()
    with client.start_generation(GenerationStart(model=ModelRef(provider="openai", name="gpt-4o"))) as rec:
        # Where a real model call would go.
        answer = f"answer to {question}"
        rec.set_result(
            input=[user_text_message(question)],
            output=[assistant_text_message(answer)],
        )
    return answer


@env.task(report=True, links=links)
async def agent(question: str = "What is Flyte?") -> str:
    return await ask(question)


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(agent, question="What is Flyte?").url)
