# /// script
# requires-python = ">=3.10"
# dependencies = ["flyte"]
# ///
"""Remote builder BYOI: two teams, two Flyte-unaware images, Flyte fills the gaps.

Each team owns their base image — built with their preferred package manager,
no knowledge of Flyte. The Flyte engineer adapts by installing flyte via the
right pip, fixing PATH, and setting PYTHONPATH.

  data_prep:  continuumio/miniconda3  — conda env, python at /opt/conda/bin/python
  training:   python:3.10-slim        — venv at /opt/venv, python at /opt/venv/bin/python

Build and push base images first (from v2_guide/remote_builder/):
  docker build -f data_prep/Dockerfile -t <your-registry>/data-prep-base:latest data_prep/
  docker build -f training/Dockerfile  -t <your-registry>/training-base:latest  training/
  docker push <your-registry>/data-prep-base:latest
  docker push <your-registry>/training-base:latest

Run (from v2_guide/remote_builder/):
  uv run main.py
"""

import pathlib

import flyte

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    flyte.init_from_config(root_dir=HERE)

    # Development: fast code iteration — image only rebuilds when base image changes.
    # run = flyte.with_runcontext(version="dev").run(prepare, raw="Hello World")
    # print(run.url)
    # run.wait()

    # Production: bake code into both images, pin to a version.
    from tasks.envs import env_data, env_train

    flyte.deploy(env_data, copy_style="none", version="3.0.0")
    flyte.deploy(env_train, copy_style="none", version="3.0.0")
