"""Vanilla ML training pipeline — linear stages, fails on the last step.

Implements the workflow shown in ``README.md``:

    ingest → preprocess & featurize → train / tune → evaluate & export (crash)

Each stage is a separate Flyte task. Earlier steps complete successfully; the
export step raises after the expensive upstream work is done — the failure mode
the self-healing MLE agents in this directory are built to recover from.

Run locally::

    python examples/genai/mle_agent/mle_pipeline.py

Run remotely::

    flyte run --follow examples/genai/mle_agent/mle_pipeline.py mle_pipeline \\
        --raw_data path/to/data.csv
"""

from __future__ import annotations

import asyncio
import pickle
import tempfile
from pathlib import Path

import flyte
from flyte.io import File

# README wall-clock labels (illustrative — not enforced unless simulate_delays=True).
README_STAGE_DURATIONS = {
    "ingest": "12 min",
    "preprocess": "38 min",
    "train": "5h 42m",
    "export": "5 min",
}

# Short sleeps when simulate_delays=True so local runs still show sequential blocking.
SIMULATED_DELAY_SECONDS = {
    "ingest": 1,
    "preprocess": 2,
    "train": 3,
    "export": 1,
}

env = flyte.TaskEnvironment(
    "mle-pipeline",
    image=flyte.Image.from_debian_base(name="mle-pipeline-image").with_pip_packages(
        "pandas",
        "numpy",
        "scikit-learn",
    ),
)


async def _simulate_stage(stage: str, simulate_delays: bool) -> None:
    print(f"[{stage}] README wall-clock: {README_STAGE_DURATIONS[stage]}")
    if simulate_delays:
        await asyncio.sleep(SIMULATED_DELAY_SECONDS[stage])


@env.task
async def ingest_data(raw_data: File, simulate_delays: bool = False) -> File:
    """Load and validate the raw CSV."""
    import pandas as pd

    await _simulate_stage("ingest", simulate_delays)

    local_path = await raw_data.download()
    df = pd.read_csv(local_path)
    required = {"feature1", "feature2", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    print(f"[ingest] loaded {len(df):,} rows from {raw_data.name}")
    return raw_data


@env.task
async def preprocess_and_featurize(data: File, simulate_delays: bool = False) -> File:
    """Clean rows and write a featurized dataset for training."""
    import pandas as pd

    await _simulate_stage("preprocess", simulate_delays)

    local_path = await data.download()
    df = pd.read_csv(local_path)
    df = df.dropna()
    df["feature_ratio"] = df["feature2"] / df["feature1"].replace(0, 1)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        out_path = f.name

    print(f"[preprocess] wrote {len(df):,} featurized rows")
    return await File.from_local(out_path)


@env.task
async def train_model(dataset: File, simulate_delays: bool = False) -> File:
    """Fit a regression model on the featurized data."""
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    await _simulate_stage("train", simulate_delays)

    local_path = await dataset.download()
    df = pd.read_csv(local_path)
    features = ["feature1", "feature2", "feature_ratio"]
    model = LinearRegression()
    model.fit(df[features], df["target"])

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(model, f)
        model_path = f.name

    print(f"[train] fitted LinearRegression on {len(df):,} rows")
    return await File.from_local(model_path)


@env.task
async def evaluate_and_export_model(
    model: File,
    dataset: File,
    simulate_delays: bool = False,
) -> File:
    """Score the model and write the production artifact — intentionally fails."""
    import pandas as pd

    await _simulate_stage("export", simulate_delays)

    model_path = await model.download()
    data_path = await dataset.download()

    fitted_model = pickle.loads(Path(model_path).read_bytes())

    df = pd.read_csv(data_path)
    features = ["feature1", "feature2", "feature_ratio"]
    predictions = fitted_model.predict(df[features])
    mae = float((abs(predictions - df["target"])).mean())
    print(f"[export] validation MAE: {mae:.6f}")

    # Deliberate bug: export path is never created, so the job dies at the last step.
    export_dir = Path("/nonexistent/production/models")
    export_path = export_dir / "model.pkl"
    export_path.write_bytes(Path(model_path).read_bytes())

    return await File.from_local(str(export_path))


@env.task
async def mle_pipeline(raw_data: File, simulate_delays: bool = False) -> str:
    """Run the full linear ML pipeline (fails on export)."""
    ingested = await ingest_data(raw_data=raw_data, simulate_delays=simulate_delays)
    featurized = await preprocess_and_featurize(data=ingested, simulate_delays=simulate_delays)
    trained = await train_model(dataset=featurized, simulate_delays=simulate_delays)
    exported = await evaluate_and_export_model(
        model=trained,
        dataset=featurized,
        simulate_delays=simulate_delays,
    )
    return await exported.download()


if __name__ == "__main__":
    import asyncio

    flyte.init_from_config()

    data_csv = Path(__file__).parent / "data.csv"
    if not data_csv.exists():
        data_csv.write_text("feature1,feature2,target\n")
        with data_csv.open("a") as f:
            for i in range(10_000):
                f.write(f"{i},{i * 2},{i * 3}\n")

    async def main() -> None:
        raw = await File.from_local(str(data_csv))
        run = flyte.run(
            mle_pipeline,
            raw_data=raw,
            simulate_delays=True,
        )
        print(f"View at: {run.url}")
        try:
            run.wait()
            print(f"Result: {run.outputs()}")
        except Exception as exc:
            print(f"Pipeline failed (expected on export step). Upstream stages completed; error: {exc}")

    asyncio.run(main())
