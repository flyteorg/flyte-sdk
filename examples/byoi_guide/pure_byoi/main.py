# /// script
# requires-python = ">=3.10"
# dependencies = ["flyte"]
# ///
"""Pure BYOI entry point.

workflow_code/ is baked into both images via their Dockerfiles.
Flyte runs each container as a black box — no code bundle is sent.

Build and push both images first (from v2_guide/pure_byoi/):
  docker build -f data_prep/Dockerfile -t <your-registry>/data-prep:latest .
  docker build -f training/Dockerfile  -t <your-registry>/training:latest  .
  docker push <your-registry>/data-prep:latest
  docker push <your-registry>/training:latest

Run (from v2_guide/pure_byoi/):
  uv run main.py
"""

from workflow_code.tasks import prepare

import flyte

DATA_PREP_IMAGE = "<your-registry>/data-prep:latest"
TRAINING_IMAGE = "<your-registry>/training:latest"

if __name__ == "__main__":
    # No root_dir — Flyte does not inject code. The images contain everything.
    flyte.init_from_config(
        project="flytesnacks",
        domain="development",
        images=(
            f"data-prep={DATA_PREP_IMAGE}",
            f"training={TRAINING_IMAGE}",
        ),
    )

    # Development: run the pipeline.
    # copy_style="none": no code bundle, the image IS the deployment.
    run = flyte.with_runcontext(copy_style="none", version="dev").run(prepare, raw="Hello World")
    print(run.url)
    run.wait()

    # Production: register task environments against the cluster.
    # from workflow_code.envs import env_data, env_train
    # flyte.deploy(env_data, copy_style="none", version="1.0.0")
    # flyte.deploy(env_train, copy_style="none", version="1.0.0")
