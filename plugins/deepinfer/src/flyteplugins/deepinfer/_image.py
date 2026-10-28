"""Container image for running DeepInfer in a Flyte task.

DeepInfer is **not distributed as a PyPI wheel**: it is built from source on Python 3.12
with a CUDA 13 toolchain and GCC >= 13. `deepinfer_image` covers both ways teams get it:

- `deepinfer_image()` builds DeepInfer from source, at a pinned commit, on NVIDIA's
  official `nvidia/cuda:<version>-devel-ubuntu24.04` image. That image already ships
  the CUDA toolkit (nvcc + headers) for both amd64 and arm64 (e.g. GB200 Grace hosts),
  and Ubuntu 24.04's default compiler is GCC 13, so nothing CUDA-specific is installed
  by hand. The build is long; build it once and reuse the image.
- `deepinfer_image(base="registry/deepinfer:tag")` layers only this plugin onto an
  image that already contains a built `deepinfer` (e.g. one your platform team bakes).

The result is a regular `flyte.Image`, so add anything else with the usual `with_*`
methods, e.g. `.with_pip_packages("unionai-reuse")` for reusable environments.
"""

from __future__ import annotations

from typing import Optional

import flyte
from flyte import Image

from flyteplugins.deepinfer import _constants as const

# DeepInfer has no tagged releases, so builds pin a commit for reproducibility. Bump
# deliberately (and re-test) rather than tracking the moving default branch.
_DEFAULT_DEEPINFER_COMMIT = "553741557a57df44a46fb3ab72b6aafeaf945337"  # main, 2026-10-25
_DEFAULT_DEEPINFER_REPO = "https://github.com/deepseek-ai/DeepInfer.git"

# Matches DeepInfer's `requirements-nvidia.txt` (torch `+cu132`, i.e. CUDA 13.2).
_DEFAULT_CUDA_VERSION = "13.2.1"

# SM100 (B200 / GB200). The image builder has no GPU, so the CUDA extensions are
# compiled for an explicit architecture instead of the one detected at build time.
_DEFAULT_CUDA_ARCH = "10.0"


def deepinfer_image(
    *,
    base: Optional[str] = None,
    python: Optional[str] = None,
    name: str = "deepinfer",
    commit: str = _DEFAULT_DEEPINFER_COMMIT,
    repo: str = _DEFAULT_DEEPINFER_REPO,
    cuda_version: str = _DEFAULT_CUDA_VERSION,
    cuda_arch: str = _DEFAULT_CUDA_ARCH,
) -> Image:
    """A Flyte image that can run DeepInfer tasks.

    Args:
        base: An image that already contains a built `deepinfer` package. When set,
            only this plugin is layered on and the build-related arguments are ignored.
        python: With `base`, the interpreter whose environment contains `deepinfer`
            (e.g. `"/opt/conda/bin/python"`). The plugin is installed into that
            environment. Unset, Flyte creates its own Python 3.12 virtualenv, which only
            sees `deepinfer` if it was installed there.
        name: Name of the resulting image.
        commit: DeepInfer git commit (or tag) to build.
        repo: Git URL to clone DeepInfer from, e.g. an internal mirror or fork.
        cuda_version: CUDA toolkit version; selects the `nvidia/cuda:<version>-devel-ubuntu24.04` base.
        cuda_arch: GPU architecture(s) the CUDA extensions are compiled for
            (`TORCH_CUDA_ARCH_LIST`).
    """
    if base is not None:
        image = flyte.Image.from_base(base).clone(name=name, python_version=(3, 12), extendable=True)
        if python is not None:
            image = image.with_env_vars({"UV_PYTHON": python})
        return image.with_pip_packages("flyteplugins-deepinfer", pre=True)

    install_dir = "/opt/deepinfer"
    return (
        flyte.Image.from_base(f"nvidia/cuda:{cuda_version}-devel-ubuntu24.04")
        .clone(name=name, python_version=(3, 12), extendable=True)
        # Ubuntu 24.04's own Python 3.12, with headers for the extension builds. Installing
        # it pulls in tzdata, which prompts for a timezone unless apt is non-interactive.
        .with_env_vars({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .with_apt_packages(
            "python3.12",
            "python3.12-dev",
            "git",
            "build-essential",
            "ninja-build",
            "pkg-config",
            "libnuma-dev",
            # DeepInfer's vendored DeepSelect probes the platform with `lspci` at import
            # time, before `DEEP_SELECT_BUILD_TARGET_PLATFORM` selects the build target.
            "pciutils",
        )
        # DeepInfer's build.sh calls `pip`, which a uv-created venv does not include.
        .with_pip_packages("pip", "ninja", "pybind11")
        .with_env_vars({const.DEEPINFER_ACCELERATOR: const.ACCELERATOR_NVIDIA, "TORCH_CUDA_ARCH_LIST": cuda_arch})
        .with_commands(
            [
                # Fetch exactly one commit; build.sh initializes the vendored submodules itself.
                f"git init {install_dir} && cd {install_dir} && git remote add origin {repo}"
                f" && git fetch --depth 1 origin {commit} && git checkout FETCH_HEAD",
                # build.sh calls `pip`; run it inside the image's virtualenv explicitly, since
                # command layers do not necessarily have the venv on PATH.
                f"cd {install_dir} && . /opt/venv/bin/activate && ./build.sh",
            ]
        )
        .with_pip_packages("flyteplugins-deepinfer", pre=True)
    )


__all__ = ["deepinfer_image"]
