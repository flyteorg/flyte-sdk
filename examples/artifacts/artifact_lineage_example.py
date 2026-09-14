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
2. **Fallback labels** — a task that resolves an artifact's value without
   binding it as a typed input (e.g. it only takes a plain string), or an app
   that reads an artifact at startup, leaves nothing for the platform to see.
   For those, the caller stamps a pair of private labels —
   `flyte.with_runcontext(labels=...)` for a run, `AppEnvironment(labels=...)`
   for an app — that the dashboard scans for. `audit_model` and
   `model_server` below both need this.

Try it:

    python examples/artifacts/artifact_lineage_example.py

then open the printed dashboard URL, or `<endpoint>/lineage` if you deploy it
separately.
"""

import tempfile
import time

from lineage import LABEL_UPSTREAM_ARTIFACT_NAME, LABEL_UPSTREAM_ARTIFACT_VERSION, ArtifactLineageAppEnvironment

import flyte
import flyte.artifacts as artifacts
from flyte.app import AppEnvironment
from flyte.io import File


def _wait_for_artifact(name: str, *, retries: int = 15, delay_s: float = 2.0):
    """`Artifact.get` right after a run completes can race artifact indexing; retry briefly."""
    from flyte.remote import Artifact

    for attempt in range(retries):
        try:
            return Artifact.get(name)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s)


def _run_and_wait(run):
    """`.wait()` alone doesn't raise on failure -- surface it immediately instead of
    letting a later `_wait_for_artifact` call fail with a confusing "not found"."""
    run.wait()
    if run.phase != "succeeded":
        raise RuntimeError(f"{run.url} finished in phase {run.phase!r}")
    return run


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


# Fallback labels: `model_ref` is a plain string, not a bound `Artifact`/`File` — the
# platform has no way to know this run consumed anything. The caller (see `run_pipeline`)
# stamps the labels explicitly so the dashboard can still find it.
@env.task
async def audit_model(model_ref: str) -> str:
    return f"audited {model_ref}"


# Apps can't bind artifacts as typed inputs at all, so the labels are the *only* way to
# declare "this app serves artifact X". The label values are only known once the model
# exists, so they're set on this module-level object (not a `clone_with()` copy -- a
# clone made inside `run_pipeline` would be a local variable, and the app resolver needs
# to find the deployed `AppEnvironment` by name in this module's *global* namespace) once
# `run_pipeline` runs, before `flyte.serve()`.
model_server = AppEnvironment(
    name="artifact-lineage-model-server",
    image=env.image,
    command="python -m http.server 8080",
)

lineage_dashboard = ArtifactLineageAppEnvironment(
    name="artifact-lineage-dashboard",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    # `train_model` is discovered via the automatic bound-input scan; `audit_model` is
    # discovered via the labels (listed here too so the bound-input scan also covers it).
    watched_tasks=["artifact_lineage_example.train_model", "artifact_lineage_example.audit_model"],
    watched_apps=[model_server.name],
)


def run_pipeline() -> None:
    """Produce, train, and audit, exercising both lineage-detection signals.

    Assumes `flyte.init_from_config()` has already run. Does *not* touch
    `lineage_dashboard` — callers that want the dashboard too (e.g. this
    module's own `__main__`, or a driver combining several example pipelines)
    deploy it separately, so multiple pipelines never fight over redeploying
    the same shared app with different `watched_tasks`/`watched_apps`.
    """
    data_run = _run_and_wait(flyte.run(produce_raw_data))
    print(data_run.url)
    data = _wait_for_artifact("raw-data")

    model_run = _run_and_wait(flyte.run(train_model, data=data))
    print(model_run.url)
    model = _wait_for_artifact("trained-model")

    # Fallback path #1 (run): stamp the labels before launching, since `audit_model`
    # only takes a plain string and the platform can't infer the artifact linkage.
    audit_labels = {LABEL_UPSTREAM_ARTIFACT_NAME: model.name, LABEL_UPSTREAM_ARTIFACT_VERSION: model.version}
    audit_run = _run_and_wait(flyte.with_runcontext(labels=audit_labels).run(audit_model, model_ref=model.tracker))
    print(audit_run.url)

    # Fallback path #2 (app): redeploy the model server with the labels now that the
    # model artifact's identity is known.
    model_server.labels = audit_labels
    flyte.serve(model_server)


if __name__ == "__main__":
    flyte.init_from_config()
    run_pipeline()

    handle = flyte.serve(lineage_dashboard)
    print(f"Lineage dashboard: {handle.url}/lineage")
    print(f"Lineage graph for trained-model: {handle.url}/lineage/artifact/trained-model")
