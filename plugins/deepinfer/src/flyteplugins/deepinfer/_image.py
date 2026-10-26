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

from typing import Any, Optional

import flyte
from flyte import Image

from flyteplugins.deepinfer import _constants as const

# NVIDIA's apt repository keyring, which provides the `cuda-toolkit-<major>-<minor>` packages.
_CUDA_KEYRING_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
)

# Defaults match DeepInfer's `requirements-nvidia.txt` (torch `+cu132`) and `build.sh`.
_DEFAULT_CUDA_VERSION = "13.0"
_DEFAULT_TORCH_VERSION = "2.13.0+cu132"
_DEFAULT_DEEPINFER_REPO = "https://github.com/deepseek-ai/DeepInfer.git"


def default_deepinfer_image(
    *,
    name: str = "deepinfer-task-image",
    python_version: tuple[int, int] = (3, 12),
    cuda_version: str = _DEFAULT_CUDA_VERSION,
    torch_version: str = _DEFAULT_TORCH_VERSION,
) -> Image:
    """A Flyte image with the plugin and the toolchain needed to build/run DeepInfer.

    Builds on the Debian flyte base, adds the NVIDIA CUDA toolkit (from NVIDIA's apt
    repo), GCC 13 (required by the C++20 extension build), the matching CUDA PyTorch
    wheel, and this plugin. The `DEEPINFER_ACCELERATOR` environment variable is set to
    `nvidia`. The defaults match what DeepInfer's own build expects.

    The DeepInfer `deepinfer` package itself is **not** compiled here (that is a
    long, hardware-specific build); layer it on with `deepinfer_image_from_source` or
    start from an image your team already baked. See the module docstring.

    Args:
        name: Image name/tag.
        python_version: Python version of the base image. DeepInfer requires 3.12.
        cuda_version: CUDA toolkit version as `"<major>.<minor>"`, installed from
            NVIDIA's apt repo to `/usr/local/cuda-<version>`.
        torch_version: PyTorch version, including its CUDA local tag (e.g.
            `"2.13.0+cu132"`). The tag selects the PyTorch wheel index.
    """
    cuda_home = f"/usr/local/cuda-{cuda_version}"
    cuda_toolkit = f"cuda-toolkit-{cuda_version.replace('.', '-')}"
    torch_cuda_tag = torch_version.partition("+")[2]
    torch_index_url = f"https://download.pytorch.org/whl/{torch_cuda_tag}/" if torch_cuda_tag else None
    return (
        flyte.Image.from_debian_base(name=name, python_version=python_version)
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
                f"wget -O /tmp/cuda-keyring.deb {_CUDA_KEYRING_URL}",
                "dpkg -i /tmp/cuda-keyring.deb",
                "apt-get update",
                f"apt-get install -y {cuda_toolkit}",
            ]
        )
        .with_pip_packages("ninja", "pybind11")
        .with_pip_packages(f"torch=={torch_version}", index_url=torch_index_url)
        .with_pip_packages("flyteplugins-deepinfer", pre=True)
        .with_env_vars(
            {
                const.DEEPINFER_ACCELERATOR: const.ACCELERATOR_NVIDIA,
                "CUDA_HOME": cuda_home,
                "PATH": f"{cuda_home}/bin:$PATH",
            }
        )
    )


def deepinfer_image_from_source(
    *,
    repo: str = _DEFAULT_DEEPINFER_REPO,
    ref: Optional[str] = None,
    install_dir: str = "/opt/deepinfer",
    **image_kwargs: Any,
) -> Image:
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
        repo: Git URL to clone DeepInfer from (e.g. an internal fork or mirror).
        ref: Branch or tag to build. Defaults to the repository's default branch; pin
            a tag for reproducible images.
        install_dir: Where the repository is cloned and built inside the image.
        **image_kwargs: Forwarded to `default_deepinfer_image` (`name`,
            `python_version`, `cuda_version`, `torch_version`).
    """
    branch = f" --branch {ref}" if ref else ""
    return default_deepinfer_image(**image_kwargs).with_commands(
        [
            f"git clone --depth 1{branch} {repo} {install_dir}",
            f"cd {install_dir} && ./build.sh",
        ]
    )


__all__ = ["deepinfer_image_from_source", "default_deepinfer_image"]
