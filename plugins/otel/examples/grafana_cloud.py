"""Exporting to a real backend, using Grafana Cloud as the worked example.

The endpoint and credentials are ordinary OTLP settings, so the same shape works for any
OTLP backend: swap the gateway URL and the auth header.

Credentials belong in a flyte.Secret rather than the source. Both the endpoint and the auth
header come from the OTLP section of the Grafana Cloud portal.

    python grafana_cloud.py            # on the cluster in your flyte config
    python grafana_cloud.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte

from flyteplugins.otel import init
from flyteplugins.otel.grafana import GrafanaTrace

# Where the Grafana UI lives and which Tempo datasource to query. The stack URL is the host
# you browse Grafana on and the datasource UID is the last path segment of
# /connections/datasources/edit/<uid>. Both are only needed for the UI link,
# so the task still runs and exports without them.
GRAFANA_HOST = os.environ.get("GRAFANA_HOST", "")
TEMPO_DATASOURCE_UID = os.environ.get("TEMPO_DATASOURCE_UID", "grafanacloud-traces")

links = (GrafanaTrace(host=GRAFANA_HOST, datasource_uid=TEMPO_DATASOURCE_UID),) if GRAFANA_HOST else ()

image = (
    flyte.Image.from_debian_base()
    # Bake the locally-built plugin wheels from ./dist so a remote run exercises the working
    # tree rather than PyPI releases. Build them with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(["flyteplugins-otel"])
)

env = flyte.TaskEnvironment(
    name="otel_grafana",
    image=image,
    secrets=[
        flyte.Secret(key="otlp_endpoint", as_env_var="OTEL_EXPORTER_OTLP_ENDPOINT"),
        flyte.Secret(key="otlp_headers", as_env_var="OTEL_EXPORTER_OTLP_HEADERS"),
    ],
)


# With the two OTEL_ variables set, init needs no arguments at all. Passing them explicitly:
#   init(
#       service_name="my-service",
#       endpoint="https://otlp-gateway-<zone>.grafana.net/otlp",
#       headers={"Authorization": "Basic <base64 instance_id:token>"},
#   )
# A base gateway URL is fine; the /v1/traces path is appended for you.
init(service_name="my-service")


@flyte.trace
async def step(i: int) -> int:
    return i + 1


# The link is rendered on the action in the Flyte UI and runs a TraceQL query for this run,
# so you land on its spans instead of hunting for the run name in Explore.
@env.task(links=links)
async def main(n: int = 3) -> int:
    total = 0
    for i in range(n):
        total += await step(i)
    return total


if __name__ == "__main__":
    if "--local" in sys.argv and not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        raise SystemExit("set OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS first")

    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(main, n=3).url)
