"""An unpartitioned artifact (owner: ML Platform).

`churn_thresholds` has no partitions: there is one current value, and every reader gets its latest version.
`calibrate` reads no artifact, only constants, so a materialization overrides one with
`--input ml-thresholds.calibrate.cutoff=0.7`.

    flyte deploy --root-dir . ml/thresholds.py env
"""

from __future__ import annotations

import json
import tempfile

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

env = flyte.TaskEnvironment(
    name="ml-thresholds",
    image=flyte.Image.from_debian_base(),
    labels={"team": "ml"},
)

churn_thresholds = artifacts.Artifact(
    "churn_thresholds",
    type=File,
    kind="generic",
    description="Alerting cutoffs for churn scores; one current version, no partitions.",
)


@env.task(produces_artifacts=(churn_thresholds,))
async def calibrate(cutoff: float = 0.5, min_users: int = 10) -> File:
    """Publish the alerting cutoffs."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"cutoff": cutoff, "min_users": min_users}, f)
    return await File.from_local(f.name)
