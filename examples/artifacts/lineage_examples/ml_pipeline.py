"""ML use case: dataset -> train -> evaluate -> serve.

A four-hop lineage chain with a merge point: `evaluate_model` consumes *two*
upstream artifacts (the model it's grading and the dataset it grades against),
so its node in the graph has two parents instead of one. The "model" is a
closed-form linear regression (slope + intercept over a handful of points) —
no ML framework needed, so the image stays tiny and the run is instant.

`ml_model_server` demonstrates the app-consumption fallback: it fetches the
model artifact directly by name at request time (a realistic pattern for a
serving app), which leaves no bound-input trace for the platform to see, so
the `upstream-artifact-name`/`upstream-artifact-version` labels are
what make it show up in the dashboard.

Try it:

    python examples/artifacts/lineage_examples/ml_pipeline.py
"""

import json
import tempfile
import time

import fastapi
from lineage import LABEL_UPSTREAM_ARTIFACT_NAME, LABEL_UPSTREAM_ARTIFACT_VERSION

import flyte
import flyte.artifacts as artifacts
from flyte.app.extras import FastAPIAppEnvironment
from flyte.io import File

# Shared by the task env and the serving app below: the module imports `fastapi` at top
# level (for `ml_app`), so any task defined here needs it too, even ones that never touch
# FastAPI themselves -- task loading imports the whole module.
image = flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn")
env = flyte.TaskEnvironment(name="ml_pipeline", image=image)


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


@env.task(produces_artifacts=True)
async def produce_dataset() -> File:
    points = [(1.0, 2.1), (2.0, 4.0), (3.0, 5.9), (4.0, 8.1), (5.0, 9.9)]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("x,y\n")
        f.writelines(f"{x},{y}\n" for x, y in points)
    file = await File.from_local(f.name)
    return artifacts.new(file, artifacts.Metadata(name="ml-dataset", description="Toy (x, y) points for regression"))


@env.task(produces_artifacts=True)
async def train_model(dataset: File) -> File:
    async with dataset.open("rb") as fh:
        rows = bytes(await fh.read()).decode().splitlines()[1:]
    points = [tuple(map(float, row.split(","))) for row in rows]
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sum((x - mean_x) ** 2 for x, _ in points)
    intercept = mean_y - slope * mean_x

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"slope": slope, "intercept": intercept}, f)
    model_file = await File.from_local(f.name)
    return artifacts.new(model_file, artifacts.Metadata(name="ml-model", description="Fitted slope/intercept"))


# Merge point: two upstream artifacts feed one run. The dashboard finds both edges
# via the automatic bound-input scan -- no labels needed for this hop.
@env.task(produces_artifacts=True)
async def evaluate_model(model: File, dataset: File) -> File:
    async with model.open("rb") as fh:
        params = json.loads(bytes(await fh.read()))
    async with dataset.open("rb") as fh:
        rows = bytes(await fh.read()).decode().splitlines()[1:]
    points = [tuple(map(float, row.split(","))) for row in rows]
    mse = sum((params["slope"] * x + params["intercept"] - y) ** 2 for x, y in points) / len(points)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"mse": mse, "n": len(points)}, f)
    report_file = await File.from_local(f.name)
    return artifacts.new(report_file, artifacts.Metadata(name="ml-eval-report", description="MSE against the dataset"))


ml_app = fastapi.FastAPI(title="ML Model Server")


@ml_app.get("/predict")
async def predict(x: float) -> dict:
    import json as _json

    from flyte.io import File as _File
    from flyte.remote import Artifact

    model_artifact = await Artifact.get.aio(name="ml-model")
    model_file = await model_artifact.to_python(_File)
    async with model_file.open("rb") as fh:
        params = _json.loads(bytes(await fh.read()))
    return {"x": x, "y_pred": params["slope"] * x + params["intercept"]}


# Fetches the model by name at request time rather than binding it as a typed input, so
# the platform sees no artifact linkage here -- the labels (set on this module-level
# object in `run_pipeline` below, once the model's identity is known) are what make this
# app discoverable as a consumer. Setting `.labels` in place (not `clone_with()`, which
# would return a local-variable copy the app resolver can't find by name in this module)
# keeps the object the resolver looks up by name in this module's global namespace.
ml_model_server = FastAPIAppEnvironment(name="ml-model-server", app=ml_app, image=image)


def run_pipeline() -> None:
    """Produce the dataset, train, evaluate, and serve. Assumes init_from_config() ran."""
    dataset_run = _run_and_wait(flyte.run(produce_dataset))
    print(dataset_run.url)
    dataset = _wait_for_artifact("ml-dataset")

    train_run = _run_and_wait(flyte.run(train_model, dataset=dataset))
    print(train_run.url)
    model = _wait_for_artifact("ml-model")

    eval_run = _run_and_wait(flyte.run(evaluate_model, model=model, dataset=dataset))
    print(eval_run.url)

    ml_model_server.labels = {LABEL_UPSTREAM_ARTIFACT_NAME: model.name, LABEL_UPSTREAM_ARTIFACT_VERSION: model.version}
    flyte.serve(ml_model_server)


if __name__ == "__main__":
    flyte.init_from_config()
    run_pipeline()
