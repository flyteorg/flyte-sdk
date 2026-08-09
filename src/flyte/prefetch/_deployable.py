"""
A prefetch task that can be registered once and launched by anything.

`hf_model` builds its environment at call time and ships the task as a
cloudpickle bundle, which means only a Python process holding the SDK can start
a prefetch. That is the right shape for a notebook or a CLI, and the wrong shape
for a control plane: a console button has no Python to run.

So this module declares the same work as an ordinary registered task. Deployed
once into a system project, it is addressable by name and version like any other
task, and a backend can bind it to a run in the user's own project -- which is
how the remote image builder's `build-image` task already works.

Both paths converge on `store_hf_model_task`, so there is one implementation of
prefetching and two ways to reach it.

Not imported from `flyte.prefetch`'s `__init__`: importing prefetch eagerly at
the `flyte` level has been a problem before, and this module is only ever
imported by the deploy driver.
"""

from __future__ import annotations

import os

import flyte
from flyte.io import Dir

from ._hf_model import HF_DOWNLOAD_IMAGE_PACKAGES, HuggingFaceModelInfo, store_hf_model_task

#: Pin the serving image at deploy time rather than rebuilding it, mirroring how
#: the image-builder task takes UNION_IMAGE_NAME_PREFIX / UNION_IMAGE_TAG from
#: the environment. Set to a fully-qualified image to skip the image build
#: entirely -- which is also the only practical way to deploy this against a
#: locally-built SDK, since the default base installs the *published* flyte.
_IMAGE_OVERRIDE = os.environ.get("FLYTE_PREFETCH_IMAGE")

#: Name of the secret holding a HuggingFace token, attached to every run of this
#: task. Empty by default, and deliberately so: public repos need no token, and
#: naming a secret that does not exist wedges the pod in
#: CreateContainerConfigError with no HuggingFace error to read -- an opaque
#: failure in exactly the first-run case. Callers that need a gated repo supply
#: the secret per run instead.
_HF_TOKEN_KEY = os.environ.get("FLYTE_PREFETCH_HF_TOKEN_KEY", "")

#: Unsharded only. Sharding pulls in vLLM and a CUDA toolchain, and needs GPUs
#: at prefetch time to run the engine that writes the per-rank files; that
#: belongs in a separate environment with its own resources, not behind a flag
#: on this one.
image = _IMAGE_OVERRIDE or flyte.Image.from_debian_base(name="prefetch-hf-model-image").with_pip_packages(
    *HF_DOWNLOAD_IMAGE_PACKAGES
)

prefetch_env = flyte.TaskEnvironment(
    name="prefetch",
    image=image,
    # The unsharded path streams HuggingFace straight to object storage in
    # chunks and never lands the weights on disk, so the disk request covers the
    # snapshot-download fallback rather than the happy path.
    resources=flyte.Resources(cpu="2", memory="8Gi", disk="50Gi"),
    secrets=[flyte.Secret(key=_HF_TOKEN_KEY, as_env_var="HF_TOKEN")] if _HF_TOKEN_KEY else None,
)


@prefetch_env.task(report=True, produces_artifacts=True)
def hf_model(repo: str, artifact_name: str = "", short_description: str = "") -> Dir:
    """
    Prefetch a HuggingFace model into a model artifact.

    Registered as ``prefetch.hf_model`` -- ``TaskEnvironment`` prefixes task
    names with the environment's name.

    Flat strings with ``""`` for unset, rather than ``str | None``: an optional
    becomes a union literal, which is awkward to construct from a non-Python
    caller, and every input has to be sent explicitly anyway because CreateRun
    does not apply a task's declared defaults. Neither field can legitimately be
    empty, so ``""`` is unambiguous.

    The remaining fields of ``HuggingFaceModelInfo`` are intentionally absent:
    architecture, model type, task and modality are all read from the model's
    own config, so asking a caller for them invites a wrong answer.

    Args:
        repo: HuggingFace repository id, e.g. ``Qwen/Qwen3-0.6B``.
        artifact_name: Name to publish under. Empty derives it from the repo's
            last path segment with ``.`` replaced by ``-``.
        short_description: Free-text description for the artifact.

    Returns:
        The directory holding the model, published as a model artifact whose
        version is the HuggingFace commit.
    """
    info = HuggingFaceModelInfo(
        repo=repo,
        artifact_name=artifact_name or None,
        short_description=short_description or None,
    )
    # Called as a plain function, not launched as a task: this already *is* the
    # task, and running it as one would nest a run inside a run for no gain.
    return store_hf_model_task(info.model_dump_json())
