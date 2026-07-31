"""A durable agent that crashes, resumes and stays one trace in Grafana.

This is the part Grafana cannot see on its own. A run that crashes and resumes is several
processes over time, each with a fresh SDK. Left alone, each attempt starts its own trace,
and every step the resumed run replayed out of its durable log is missing entirely, because
replayed steps never execute and so nothing instruments them.

With the plugin, the trace id is derived from the run, so both attempts record into one
trace; replayed steps appear marked flyte.replayed; and the generations already paid for on
the first attempt are not paid for again, because flyte.trace replays them from the log.

Run it once and it fails partway. Resume the same run and watch the trace fill in.

    pip install "flyteplugins-agento11y[openai]"
    export AGENTO11Y_ENDPOINT=... AGENTO11Y_AUTH_TOKEN=...
    python durable_agent.py            # on the cluster in your flyte config
    python durable_agent.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte
from agento11y import GenerationStart, ModelRef, assistant_text_message
from flyteplugins.otel.grafana import GrafanaTrace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.agento11y import GrafanaAgentObservability, get_client, init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so the trace is visible in the task logs.

    Generations go to Grafana over their own channel regardless; this is only about where the
    spans land, and seeing the replayed steps is the whole point of this example.
    """
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


# disable_batch keeps nothing buffered, so spans recorded before a crash are already exported
# when the process dies. Batching is the better default in production.
init(service_name="durable-agent", exporter=_console_or_otlp(), disable_batch=True)

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
    name="durable",
    image=image,
    # agento11y defaults its auth mode to "none", so a token alone is never sent and the
    # export comes back 401. Grafana Cloud uses Basic with the instance id as the username,
    # which agento11y fills from AGENTO11Y_AUTH_TENANT_ID when the mode is "basic".
    env_vars={"AGENTO11Y_AUTH_MODE": "basic"},
    secrets=[
        flyte.Secret(key="sam_agento11y_endpoint", as_env_var="AGENTO11Y_ENDPOINT"),
        flyte.Secret(key="sam_agento11y_token", as_env_var="AGENTO11Y_AUTH_TOKEN"),
        flyte.Secret(key="sam_agento11y_tenant_id", as_env_var="AGENTO11Y_AUTH_TENANT_ID"),
        # Spans go to Tempo over OTLP; generations go to Agent Observability over their
        # own channel. Both are needed for the two links on the task to resolve.
        flyte.Secret(key="sam_otlp_endpoint", as_env_var="OTEL_EXPORTER_OTLP_ENDPOINT"),
        flyte.Secret(key="sam_otlp_headers", as_env_var="OTEL_EXPORTER_OTLP_HEADERS"),
    ],
)


@flyte.trace
async def step(i: int) -> str:
    """One durable model turn.

    Recorded by flyte.trace, so a resumed run replays it from the durable log rather than
    calling the model again. It still appears in the trace, marked as replayed.
    """
    with get_client().start_generation(GenerationStart(model=ModelRef(provider="openai", name="gpt-4o"))) as rec:
        rec.set_result(output=[assistant_text_message(f"thought {i}")])
    return f"thought {i}"


@env.task(retries=3, report=True, links=links)
async def agent(steps: int = 5, fail_at: int = 2) -> list[str]:
    # Crash on the first attempt only. A deterministic crash would re-raise at the same point
    # on every retry and the run could never finish, so you would see the replay but never the
    # recovery. FLYTE_ATTEMPT_NUMBER is 1-based, so the first attempt is 1.
    crash = flyte.ctx().attempt_number <= 1

    results = []
    for i in range(steps):
        results.append(await step(i))
        if crash and i == fail_at:
            raise RuntimeError(f"crashed at step {i}")
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    run = runner.run(agent, steps=5, fail_at=2)
    print(run.url)
    print("the retry replays steps 0-2 and executes the rest, all in one trace")
