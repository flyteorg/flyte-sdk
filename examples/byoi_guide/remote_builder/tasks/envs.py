import pathlib

import flyte

# Root of the remote_builder example (one level up from this file)
HERE = pathlib.Path(__file__).parent.parent

REGISTRY = "<your-registry>"  # e.g. "ghcr.io/your-org" or "123456789.dkr.ecr.us-east-1.amazonaws.com"

# ── Team B: pip venv ────────────────────────────────────────────────────────────
# Base image uses a venv at /opt/venv — python at /opt/venv/bin/python.
# The venv is not activated: PATH doesn't include /opt/venv/bin.
# Flyte adapts: installs flyte via venv's pip, adds venv to PATH, sets PYTHONPATH.
# $PATH expands at Docker build time to the base image's PATH value.
env_train_image = (
    flyte.Image.from_base("<your-registry>/training-base:latest")
    .clone(name="<your-org>/<your-image>", registry=REGISTRY, extendable=True)
    .with_commands(["/opt/venv/bin/pip install flyte"])
    .with_env_vars(
        {
            "PATH": "/opt/venv/bin:$PATH",
            "PYTHONPATH": "/workspace",  # /workspace is WORKDIR
        }
    )
    .with_code_bundle()
)

# ── Team A: conda ───────────────────────────────────────────────────────────────
# Base image uses conda — python at /opt/conda/bin/python.
# conda's own Dockerfile already adds /opt/conda/bin to PATH, so python is findable.
# Flyte adapts: installs flyte via conda's pip, sets PYTHONPATH.
env_data_image = (
    flyte.Image.from_base("<your-registry>/data-prep-base:latest")
    .clone(name="<your-org>/<your-image>", registry=REGISTRY, extendable=True)
    .with_commands(["/opt/conda/bin/pip install flyte"])
    .with_env_vars({"PYTHONPATH": "/app"})  # /app is WORKDIR; code bundle extracts here
    .with_code_bundle()
)

env_train = flyte.TaskEnvironment(name="training", image=env_train_image)
env_data = flyte.TaskEnvironment(name="data-prep", image=env_data_image, depends_on=[env_train])
