"""The consolidated lineage dashboard: watches every example pipeline at once.

Deploys a single `ArtifactLineageAppEnvironment` (reusing the
`artifact-lineage-dashboard` app name from `artifact_lineage_example.py`, so
there's exactly one dashboard running rather than one per pipeline) with
`watched_tasks`/`watched_apps` covering the mechanism walkthrough plus all
three use-case pipelines in this directory: ML (`ml_pipeline.py`), ETL
(`etl_pipeline.py`), and BI (`bi_pipeline.py`).

Try it:

    python examples/artifacts/lineage_examples/dashboard.py

This runs every pipeline first (so the dashboard has real lineage to show)
and then deploys/serves the dashboard itself.
"""

import sys
from pathlib import Path

from lineage import ArtifactLineageAppEnvironment

import flyte

# `lineage_examples` is a package (this file is a member of it), but
# `artifact_lineage_example` is a flat sibling module one directory up --
# make both importable the same way regardless of how this script was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

lineage_dashboard = ArtifactLineageAppEnvironment(
    name="artifact-lineage-dashboard",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    watched_tasks=[
        # artifact_lineage_example.py -- the mechanism walkthrough
        "artifact_lineage_example.train_model",
        "artifact_lineage_example.audit_model",
        # ml_pipeline.py -- dataset -> train -> evaluate (merge) -> serve
        "lineage_examples.ml_pipeline.train_model",
        "lineage_examples.ml_pipeline.evaluate_model",
        # etl_pipeline.py -- two sources (diamond) -> transform -> join -> load
        "lineage_examples.etl_pipeline.transform_orders",
        "lineage_examples.etl_pipeline.join_orders_customers",
        "lineage_examples.etl_pipeline.load_warehouse",
        # bi_pipeline.py -- ingest -> daily/weekly aggregates -> report (fan-in)
        "lineage_examples.bi_pipeline.aggregate_daily",
        "lineage_examples.bi_pipeline.aggregate_weekly",
        "lineage_examples.bi_pipeline.build_report",
    ],
    watched_apps=[
        "artifact-lineage-model-server",
        "ml-model-server",
        "etl-monitor-app",
        "bi-dashboard-app",
    ],
)


if __name__ == "__main__":
    flyte.init_from_config()

    import artifact_lineage_example

    from lineage_examples import bi_pipeline, etl_pipeline, ml_pipeline

    artifact_lineage_example.run_pipeline()
    ml_pipeline.run_pipeline()
    etl_pipeline.run_pipeline()
    bi_pipeline.run_pipeline()

    handle = flyte.serve(lineage_dashboard)
    print(f"Lineage dashboard: {handle.url}/lineage")
