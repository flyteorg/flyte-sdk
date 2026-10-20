"""Push: react to a new churn_model version (owner: ML Quality).

The trigger binds to the handle, extracted at deploy, so it can attach before the first version of
`churn_model` exists. Nothing is written by the SDK for the trigger's lineage: the backend derives
`lineage.consumes=churn_model` from the automation and `lineage.produces=trigger:revalidate-on-new-model`.

    flyte deploy --root-dir . triggers/revalidate.py env
"""

from __future__ import annotations

import json

from ml.train import churn_model

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(name="ml-quality", image=flyte.Image.from_debian_base(), labels={"team": "ml-quality"})

revalidate = flyte.Trigger(
    name="revalidate-on-new-model",
    automation=flyte.OnArtifact(churn_model),  # the handle, not a string
    inputs={"model": flyte.TriggeredArtifact, "threshold": 0.82},
)


@env.task(triggers=(revalidate,))
async def validate(model: File, threshold: float) -> str:
    """Sanity-check a freshly published model."""
    async with model.open("rb") as fh:
        weights = json.loads(bytes(await fh.read()))
    ok = "churned" in weights and "retained" in weights
    return f"model ok={ok} (threshold {threshold})"
