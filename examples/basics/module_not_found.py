# module_not_found.py
#
# /// script
# requires-python = "==3.13"
# dependencies = [
#    "kubernetes",
# ]
# ///
#
# Reproduces the "empty error message" problem from the support thread, with a custom
# pod template on the task environment.
#
# When a task module imports something that exists locally but is NOT in the task's
# container image, AND the import is at *module top level* (not inside the task body),
# the container crashes while Flyte is still *loading* the task -- before the error-
# capturing run wrapper exists. No `error.pb` gets written, so the backend has no error
# document to show and the task UI displays only:
#
#     [primary] terminated with exit code (1). Reason [Error]. Message:
#     .
#
# The real `ModuleNotFoundError` is buried at the very bottom of the task logs.
#
# Why top-level matters (this is the whole point of the repro):
#   - Top-level import  -> fails in `_download_and_load_task` / `load_task`, which runs
#     BEFORE `extract_download_run_upload`. `upload_error()` never runs, so no error.pb
#     is uploaded -> the UI message is empty. (This is the bug we want to surface.)
#   - In-body import    -> fails inside `convert_and_run`, which IS wrapped by the error
#     handler, so error.pb is written and the UI correctly shows "No module named ...".
#
# The `[primary]` in the UI message is the pod's primary container name, set below via the
# pod template (`primary_container_name="primary"`).
#
# To reproduce the empty message:
#   1. Install the dependency LOCALLY so `flyte run` can import this file and submit the
#      task:   pip install emoji
#   2. Do NOT add `emoji` to the image, so the container lacks it at runtime.
#   3. Run remotely:   flyte run module_not_found.py main
#   4. Observe the empty error in the UI, and the real ModuleNotFoundError only at the
#      bottom of the task logs.

# This top-level import is present locally (step 1) but missing from the task image, so
# the container dies while loading this module -- before Flyte can capture the error.
import emoji
from kubernetes.client import (
    V1Container,
    V1EnvVar,
    V1PodSpec,
)

import flyte

# A custom pod template on the environment. The primary container name shows up as
# `[primary]` in the "terminated with exit code (1)" message in the UI.
pod_template = flyte.PodTemplate(
    primary_container_name="primary",
    labels={"app": "module-not-found-repro"},
    annotations={"flyte.org/example": "module_not_found"},
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                env=[V1EnvVar(name="EXAMPLE", value="module_not_found")],
            )
        ],
    ),
)

# The default image deliberately does NOT install `emoji`. To make this pass (and prove
# the asymmetry), build the image with the dependency instead:
#
#     image=flyte.Image.from_debian_base().with_pip_packages("emoji")
#
env = flyte.TaskEnvironment(
    name="module_not_found",
    resources=flyte.Resources(memory="250Mi"),
    pod_template=pod_template,
)


@env.task
def main(name: str = "world") -> str:
    # `emoji` is used here, but the failure happens at import time above -- the container
    # never reaches this line.
    return emoji.emojize(f"Hello {name} :wave:")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.name)
    print(run.url)
    # The UI will show an empty "[primary] terminated with exit code (1)" message; the real
    # `ModuleNotFoundError: No module named 'emoji'` is only at the bottom of the task logs.
    run.wait()
