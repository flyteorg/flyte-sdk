"""Artifacts as a model registry: candidate -> evaluate -> production -> rollout.

A registry needs four things: versions of a named model, metadata on each, a way
to say which version is "production", and automation on transitions. Artifacts
give the first two directly; partitions and triggers cover the other two without
a mutable alias, which the artifact service does not have (there is no
UpdateArtifact RPC — a version's metadata is immutable once published).

The pattern:

    sweep (report=True)                   the experiment
      └── train x N (cache="auto")        each child is a tracked run: inputs, task
          │                               version, dataset version, cache hit/miss
          └── leaderboard report; winner published as   stage=candidate
                                                            │
    on_candidate: OnArtifact(MODEL, stage="candidate")      follows every deploy
      └── evaluate: beats prod? publish a pointer version   stage=production
                    (same URI, parents=[candidate])         │
    on_production: OnArtifact(MODEL, stage="production")    pinned to a released
      └── rollout: smoke-test, hand off to serving          task version

- **Stages are partitions.** `stage=candidate` and `stage=production` are string
  partitions on the same artifact name. `Artifact.get(MODEL, stage="production")`
  answers "what is prod", `OnArtifact(MODEL, stage=...)` narrows each trigger, so
  the candidate trigger never sees a promotion and vice versa.
- **Promotion is a new version, not an edit.** `evaluate` republishes the
  candidate's value into the production partition with the candidate as its
  parent: no data copy (same blob URI), and lineage records what was promoted
  from what.
- **The rollout trigger is pinned.** `Trigger(task_version="v1.4.0")` keeps the
  production entrypoint on a released version of `rollout` while CI keeps
  deploying newer ones. Promote or roll back the rollout *code* by changing the
  pin and redeploying; promote the *model* by publishing into the partition.
- **The sweep is the experiment.** Each `train` action already stores params
  (inputs), code (task version) and data (the dataset artifact version); the
  parent task holds all results, so it renders the comparison as its report and
  attaches the same table to the candidate as a card. Metrics in `attrs` are
  strings — filterable by equality, not by range — so cross-sweep comparison is a
  client-side walk of `Artifact.listall(MODEL, stage="candidate")`.

Try it:

    flyte deploy examples/artifacts/model_registry.py env
    python examples/artifacts/model_registry.py

then watch `evaluate` and `rollout` runs appear, launched by the triggers. The
`rollout` trigger pins `task_version`, so deploy once under that version first
(`flyte deploy --version v1.4.0 ...`) or set `ROLLOUT_TASK_VERSION` to a version
you have deployed.
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import flyte
import flyte.artifacts as artifacts
import flyte.report
from flyte.io import Dir, File
from flyte.remote import Artifact

MODEL = "sentiment-model"
DATASET = "reviews"

env = flyte.TaskEnvironment(name="model_registry_example")


# ------------------------------------------------------------------ experiment --
@dataclass
class TrainResult:
    lr: float
    accuracy: float
    f1: float
    weights: Dir


@env.task(cache="auto")
async def train(dataset: Artifact, lr: float) -> TrainResult:
    """One run of the experiment. Cached on (dataset version, lr): re-running the
    sweep with the same inputs does not retrain."""
    data = await dataset.to_python.aio(File)
    async with data.open("rb") as fh:
        n = len((await fh.read()).splitlines())  # pretend to train on it
    out = Path(tempfile.mkdtemp())
    (out / "weights.safetensors").write_text(f"lr={lr} rows={n}")
    (out / "config.json").write_text('{"architecture": "distilbert-base"}')
    weights = await Dir.from_local(out)
    # Fake metrics that depend on lr so the leaderboard has an order.
    accuracy = round(0.9 + lr * 10, 4)
    return TrainResult(lr=lr, accuracy=accuracy, f1=round(accuracy - 0.02, 4), weights=weights)


@env.task(report=True)
async def sweep(dataset: Artifact, lrs: list[float]) -> Artifact:
    """The experiment. Fans out one `train` action per lr, renders the comparison
    as this action's report, and publishes the winner as a candidate."""
    results = await asyncio.gather(*(train(dataset, lr) for lr in lrs))
    rows = sorted(results, key=lambda r: -r.accuracy)
    table = _leaderboard_html(rows)
    await flyte.report.log(table, do_flush=True)

    best = rows[0]
    card = await artifacts.Card.create_from.aio(content=table, format="html", card_type="model")
    return await Artifact.create.aio(
        best.weights,
        name=MODEL,
        kind="model",
        description=f"sweep winner lr={best.lr}",
        attrs={
            "framework": "torch",
            "architecture": "distilbert-base",
            "task": "text-classification",
            "serial_format": "safetensors",
            # Metrics: strings, equality-filterable only.
            "accuracy": str(best.accuracy),
            "f1": str(best.f1),
            "lr": str(best.lr),
        },
        card=card,
        partitions={"stage": "candidate", "date": date.today()},
        # Lineage across names: this model version derives from that dataset version.
        parents=[dataset.artifact_version_id],
    )


# ------------------------------------------------------------------- promotion --
on_candidate = flyte.Trigger(
    name="eval-candidate",
    # Only candidate versions; the pointer versions `evaluate` publishes never fire this.
    automation=flyte.OnArtifact(MODEL, stage="candidate"),
    inputs={"candidate": flyte.TriggeredArtifact},
    description="Evaluate every new candidate against production",
)


@env.task(triggers=[on_candidate])
async def evaluate(candidate: Artifact) -> str:
    cand_acc = float(_attrs(candidate)["accuracy"])
    try:
        prod = await Artifact.get.aio(MODEL, stage="production")
        prod_acc = float(_attrs(prod)["accuracy"])
    except Exception:  # nothing promoted yet
        prod_acc = -1.0

    if cand_acc <= prod_acc:
        return f"rejected {candidate.version}: accuracy {cand_acc} <= production {prod_acc}"

    # Promote: republish the same value into the production partition. No blob is
    # copied, and parents=[candidate] records what this pointer was promoted from.
    weights = await candidate.to_python.aio(Dir)
    promoted = await Artifact.create.aio(
        weights,
        name=MODEL,
        kind="model",
        description=f"promoted from {candidate.version}",
        attrs={**_attrs(candidate), "promoted_from": candidate.version},
        partitions={"stage": "production", "date": date.today()},
        parents=[candidate.artifact_version_id],
    )
    return f"promoted {candidate.version} -> {promoted.version}"


# --------------------------------------------------------------------- rollout --
on_production = flyte.Trigger(
    name="prod-rollout",
    # Only production versions fire this.
    automation=flyte.OnArtifact(MODEL, stage="production"),
    inputs={"model": flyte.TriggeredArtifact},
    # Stay on a released version of `rollout` while CI deploys newer ones.
    # Change the pin and redeploy to promote or roll back the rollout code.
    task_version=os.environ.get("ROLLOUT_TASK_VERSION", "v1.4.0"),
    description="Hand a promoted model to serving",
)


@env.task(triggers=[on_production])
async def rollout(model: Artifact) -> str:
    """Smoke-test the promoted weights and hand off to serving.

    In a real pipeline this is where the serving app gets redeployed with
    `flyte.app.ArtifactValue(name=MODEL, version=model.version)` as a parameter —
    see examples/ml/image_classification/serving.py. The resolved version is
    recorded on the app, so "what is production" is answerable from the app as
    well as from the `stage=production` partition.
    """
    weights = await model.to_python.aio(Dir)
    config = await weights.get_file("config.json")
    assert config is not None, "promoted model is missing config.json"
    return model.version


# --------------------------------------------------------------------- helpers --
def _attrs(a: Artifact) -> dict[str, str]:
    return dict(a.pb2.spec.info.user_metadata)


def _leaderboard_html(rows: list[TrainResult]) -> str:
    body = "".join(f"<tr><td>{r.lr}</td><td>{r.accuracy:.4f}</td><td>{r.f1:.4f}</td></tr>" for r in rows)
    return f"<h2>{MODEL} sweep</h2><table><tr><th>lr</th><th>accuracy</th><th>f1</th></tr>{body}</table>"


if __name__ == "__main__":
    flyte.init_from_config()

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("text,label\ngreat,1\nterrible,0\nfine,1\n")
    dataset = Artifact.create(
        File.from_local(f.name),
        name=DATASET,
        kind="data",
        external_ref="s3://corp-data/reviews/2026-11.csv",  # where it was imported from
    )

    run = flyte.with_runcontext(labels={"experiment": "lr-sweep"}).run(sweep, dataset=dataset, lrs=[1e-3, 3e-4, 1e-4])
    print(run.url)
    run.wait()
    # From here the triggers take over: candidate -> evaluate -> production -> rollout.

    print("candidates:")
    for a in Artifact.listall(MODEL, stage="candidate", limit=10):
        print(f"  {a.version}  accuracy={_attrs(a).get('accuracy')}  {a.source_run_url}")
