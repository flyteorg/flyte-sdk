import pathlib

import flyte
from workspace_app.tasks.etl_tasks import etl_pipeline
from workspace_app.tasks.ml_tasks import ml_pipeline

SRC_DIR = pathlib.Path(__file__).parent.parent  # -> 01_workspace_monorepo/src/


if __name__ == "__main__":
    flyte.init_from_config(root_dir=SRC_DIR)

    features = [1.5, 3.0, 4.5, 6.0, 7.5]
    labels = [0.0, 1.0, 2.0, 3.0, 4.0]

    # Development: fast deploy (code bundle delivers source at runtime)
    etl_run = flyte.run(etl_pipeline, n=10)
    print(f"ETL run: {etl_run.url}")

    ml_run = flyte.run(ml_pipeline, features=features, labels=labels)
    print(f"ML run: {ml_run.url}")

    # Production: bake source into the image (uncomment and set a version)
    # flyte.deploy(etl_env, copy_style="none", version="1.0.0")
    # flyte.deploy(ml_env, copy_style="none", version="1.0.0")
