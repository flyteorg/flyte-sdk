"""Container images for running DeepInfer in a Flyte task.

DeepInfer is **not distributed as a PyPI wheel** — it must be built from source on a
Python 3.12 environment with a CUDA 13 (NVIDIA) or CANN 9.2 (Ascend) toolchain and
GCC >= 13, against an NVIDIA SM100 GPU or Ascend 950 NPU. The helpers here produce a
workable starting image rather than a magical one:

- `default_deepinfer_image` layers the plugin and the CUDA 13 + GCC 13 build toolchain
  onto the flyte base image. It does **not** compile DeepInfer — that is a long,
  hardware-specific build best done by your platform team and baked into a base image.
- `deepinfer_image_from_source` additionally clones DeepInfer and runs its `build.sh`
  inside the image, which is the turnkey-but-heavy option (a large build, minutes+).

Most teams supply a pre-built DeepInfer image directly via `image="registry/…"` and
simply need the plugin layered on: `default_deepinfer_image().with_commands(...)` or
`Image.from_base("…").with_pip_packages("flyteplugins-deepinfer", pre=True)`.
"""

from __future__ import annotations

import flyte
from flyte import Image

from flyteplugins.deepinfer import _constants as const

# DeepInfer's build script expects this to select the NVIDIA backend.
DEEPINFER_ACCELERATOR_DEFAULT = const.ACCELERATOR_NVIDIA

# NVIDIA CUDA 13 repository + toolkit package (matching DeepInfer's `requirements-nvidia.txt`
# torch `+cu132` line and the NCCL 2.32.3 patch done in `build.sh`).
_CUDA_KEYRING_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
)
_CUDA_TOOLKIT_PACKAGE = "cuda-toolkit-13-0"
_CUDA_HOME = "/usr/local/cuda-13.0"

# DeepInfer's upstream repository, cloned by `deepinfer_image_from_source`.
_DEEPINFER_REPO = "https://github.com/deepseek-ai/DeepInfer.git"
_DEEPINFER_DIR = "/opt/deepinfer"


def default_deepinfer_image(*, name: str = "deepinfer-task-image") -> Image:
    """A Flyte image with the plugin and the toolchain needed to build/run DeepInfer.

    Builds on the Debian/Python 3.12 flyte base, adds the NVIDIA CUDA 13 toolkit
    (from NVIDIA's apt repo), GCC 13 (required by the C++20 extension build), the
    pinned CUDA 13 PyTorch wheel, and this plugin. The `DEEPINFER_ACCELERATOR`
    environment variable is set to `nvidia`.

    The DeepInfer `deepinfer` package itself is **not** compiled here (that is a
    long, hardware-specific build); layer it on with `deepinfer_image_from_source` or
    start from an image your team already baked. See the module docstring.

    Args:
        name: Image name/tag.
    """
    img = (
        flyte.Image.from_debian_base(name=name, python_version=(3, 12))
        .with_apt_packages(
            "wget",
            "curl",
            "ca-certificates",
            "gnupg",
            "build-essential",
            "gcc-13",
            "g++-13",
            "ninja-build",
            "git",
            "pkg-config",
            "libssl-dev",
            "libnuma-dev",
        )
        .with_commands(
            [
                "wget " + _CUDA_KEYRING_URL,
                "dpkg -i cuda-keyring_1.1-1_all.deb",
                "apt-get update",
                f"apt-get install -y {_CUDA_TOOLKIT_PACKAGE}",
            ]
        )
        .with_pip_packages("ninja", "pybind11")
        .with_pip_packages("torch==2.13.0+cu132", index_url="https://download.pytorch.org/whl/cu132/")
        .with_pip_packages("flyteplugins-deepinfer", pre=True)
        .with_env_vars(
            {
                const.DEEPINFER_ACCELERATOR: DEEPINFER_ACCELERATOR_DEFAULT,
                "CUDA_HOME": _CUDA_HOME,
                "PATH": f"{_CUDA_HOME}/bin:$PATH",
            }
        )
    )
    return img


def deepinfer_image_from_source(*, name: str = "deepinfer-task-image") -> Image:
    """A `default_deepinfer_image` plus DeepInfer compiled from source.

    Clones the DeepInfer repository and runs its official `build.sh` (NVIDIA
    accelerator) inside the image with the `venv`-providing Python 3.12 environment,
    so the resulting image contains a ready-to-import `deepinfer` package. Heavy:
    the build compiles flash-attn-4, tilelang, tile-kernels, and the DeepInfer CUDA
    extensions, so it takes a long time and a beefy builder.

    Because the build script `pip install -e` into the active environment, the
    commands run inside the same Python the image uses. If the build fails, prefer
    baking your platform team's prebuilt DeepInfer image and layering the plugin on.

    Args:
        name: Image name/tag.
    """
    img = default_deepinfer_image(name=name)
    img = img.with_commands(
        [
            f"git clone --depth 1 {_DEEPINFER_REPO} {_DEEPINFER_DIR}",
            f"cd {_DEEPINFER_DIR} && ./build.sh",
        ]
    )
    return img


__all__ = ["deepinfer_image_from_source", "default_deepinfer_image"]
