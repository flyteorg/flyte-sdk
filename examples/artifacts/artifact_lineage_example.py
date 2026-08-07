"""Artifact lineage: track producers and consumers of artifacts across runs and apps.

`ArtifactLineageAppEnvironment` (see `lineage/`) renders, for any published
artifact, the chain of everything connected to it: every producing run back to
the original run/artifact in the chain, and every consuming run, artifact, or
app downstream of it. Two signals feed the graph:

1. **Automatic** — when an artifact is bound as a typed task input (a plain
   `flyte.run(task, model=artifact)` call, or an `OnArtifact` trigger binding
   `flyte.TriggeredArtifact`), the platform stamps the artifact's identity on
   the input literal itself. No extra code needed; see `train_model` below,
   which consumes `raw-data` this way.
2. **Fallback label** — a task that resolves an artifact's value without
   binding it as a typed input (e.g. it only takes a plain string), or an app
   that reads an artifact at startup, leaves nothing for the platform to see.
   For those, the caller stamps a private `__upstream_artifact__` label —
   `flyte.with_runcontext(labels=...)` for a run, `AppEnvironment(labels=...)`
   for an app — that the dashboard scans for. `audit_model` and
   `model_server` below both need this.

Try it:

    python examples/artifacts/artifact_lineage_example.py

then open the printed dashboard URL, or `<endpoint>/lineage` if you deploy it
separately.
"""

import tempfile

from lineage import LABEL_UPSTREAM_ARTIFACT, ArtifactLineageAppEnvironment

import flyte
import flyte.artifacts as artifacts
from flyte.app import AppEnvironment
from flyte.io import File

env = flyte.TaskEnvironment(
    name="artifact_lineage_example",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
)


@env.task(produces_artifacts=True)
async def produce_raw_data() -> File:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("feature,label\n1,0\n2,1\n")
    file = await File.from_local(f.name)
    return artifacts.new(file, artifacts.Metadata(name="raw-data", description="Raw ingested training data"))


# Automatic detection: `data` binds the artifact as a typed `File` input, so the
# platform stamps `Literal.artifact_id` on it — the dashboard finds this edge by
# scanning `train_model`'s runs for artifact-bound inputs, no label required.
@env.task(produces_artifacts=True)
async def train_model(data: File) -> File:
    async with data.open("rb") as fh:
        rows = bytes(await fh.read()).decode().splitlines()
    with tempfile.NamedTemporaryFile("w", suffix=".bin", delete=False) as f:
        f.write(f"weights trained on {len(rows) - 1} rows")
    weights = await File.from_local(f.name)
    return artifacts.new(weights, artifacts.Metadata(name="trained-model", description="Trained model weights"))


# Fallback label: `model_ref` is a plain string, not a bound `Artifact`/`File` — the
# platform has no way to know this run consumed anything. The caller (see `__main__`)
# stamps the `__upstream_artifact__` label explicitly so the dashboard can still find it.
@env.task
async def audit_model(model_ref: str) -> str:
    return f"audited {model_ref}"


# Apps can't bind artifacts as typed inputs at all, so the label is the *only* way to
# declare "this app serves artifact X". The label value is only known once the model
# exists, so it's attached via `clone_with(labels=...)` + redeploy in `__main__`,
# not at module-definition time.
model_server = AppEnvironment(
    name="artifact-lineage-model-server",
    image=env.image,
    command="python -m http.server 8080",
)

lineage_dashboard = ArtifactLineageAppEnvironment(
    name="artifact-lineage-dashboard",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    # `train_model` is discovered via the automatic bound-input scan; `audit_model` is
    # discovered via the label (listed here too so the bound-input scan also covers it).
    watched_tasks=["artifact_lineage_example.train_model", "artifact_lineage_example.audit_model"],
    watched_apps=[model_server.name],
)


if __name__ == "__main__":
    flyte.init_from_config()

    from flyte.remote import Artifact

    data_run = flyte.run(produce_raw_data)
    print(data_run.url)
    data_run.wait()
    data = Artifact.get("raw-data")

    model_run = flyte.run(train_model, data=data)
    print(model_run.url)
    model_run.wait()
    model = Artifact.get("trained-model")

    # Fallback path #1 (run): stamp the label before launching, since `audit_model`
    # only takes a plain string and the platform can't infer the artifact linkage.
    audit_run = flyte.with_runcontext(labels={LABEL_UPSTREAM_ARTIFACT: model.tracker}).run(
        audit_model, model_ref=model.tracker
    )
    print(audit_run.url)
    audit_run.wait()

    # Fallback path #2 (app): redeploy the model server with the label now that the
    # model artifact's identity is known.
    labeled_model_server = model_server.clone_with(
        name=model_server.name, labels={LABEL_UPSTREAM_ARTIFACT: model.tracker}
    )
    flyte.serve(labeled_model_server)

    handle = flyte.serve(lineage_dashboard)
    print(f"Lineage dashboard: {handle.url}/lineage")
    print(f"Lineage graph for trained-model: {handle.url}/lineage/artifact/trained-model")
