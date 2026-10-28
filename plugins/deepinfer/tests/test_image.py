"""Unit tests for `deepinfer_image`."""

from flyteplugins.deepinfer import deepinfer_image
from flyteplugins.deepinfer._image import _DEFAULT_DEEPINFER_COMMIT


def _layers(img, kind):
    return [layer for layer in img._layers if type(layer).__name__ == kind]


def _commands(img):
    return "\n".join(c for layer in _layers(img, "Commands") for c in layer.commands)


def _pip(img):
    return [p for layer in _layers(img, "PipPackages") for p in layer.packages or ()]


def _env(img):
    return {k: v for layer in _layers(img, "Env") for k, v in layer.env_vars}


def test_source_build_on_cuda_devel_base():
    img = deepinfer_image()
    assert img.base_image == "nvidia/cuda:13.2.1-devel-ubuntu24.04"
    assert img.python_version == (3, 12)
    # No hand-rolled CUDA install: the toolkit comes with the base image.
    assert "cuda-keyring" not in _commands(img)
    assert f"git fetch --depth 1 origin {_DEFAULT_DEEPINFER_COMMIT}" in _commands(img)
    assert "./build.sh" in _commands(img)
    assert {"pip", "flyteplugins-deepinfer"} <= set(_pip(img))
    assert _env(img)["DEEPINFER_ACCELERATOR"] == "nvidia"
    assert _env(img)["TORCH_CUDA_ARCH_LIST"] == "10.0"


def test_source_build_is_configurable():
    img = deepinfer_image(commit="abc123", repo="https://git.example/deepinfer.git", cuda_version="13.4.3")
    assert img.base_image == "nvidia/cuda:13.4.3-devel-ubuntu24.04"
    assert "git remote add origin https://git.example/deepinfer.git" in _commands(img)
    assert "git fetch --depth 1 origin abc123" in _commands(img)


def test_prebuilt_base_only_layers_the_plugin():
    img = deepinfer_image(base="registry.io/deepinfer:b200", python="/opt/conda/bin/python")
    assert img.base_image == "registry.io/deepinfer:b200"
    assert _commands(img) == ""
    assert _pip(img) == ["flyteplugins-deepinfer"]
    assert _env(img) == {"UV_PYTHON": "/opt/conda/bin/python"}


def test_image_is_extendable():
    img = deepinfer_image(base="registry.io/deepinfer:b200").with_pip_packages("unionai-reuse")
    assert "unionai-reuse" in _pip(img)
