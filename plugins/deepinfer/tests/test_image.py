"""Unit tests for the DeepInfer image helpers."""

from flyteplugins.deepinfer import _constants as const
from flyteplugins.deepinfer import deepinfer_image_from_source, default_deepinfer_image


def _layers(img):
    return list(img._layers)


def test_default_image_has_plugin_and_toolchain():
    img = default_deepinfer_image()
    layers = _layers(img)

    pip = [getattr(layer, "packages", ()) or () for layer in layers if type(layer).__name__ == "PipPackages"]
    pip_flat = [p for group in pip for p in group]
    cmd_groups = [getattr(layer, "commands", ()) or () for layer in layers if type(layer).__name__ == "Commands"]
    joined_commands = "\n".join(c for group in cmd_groups for c in group)

    assert "cuda-toolkit" in joined_commands
    assert "flyteplugins-deepinfer" in pip_flat
    assert any(str(p).startswith("torch==") for p in pip_flat)

    env = {k: v for layer in layers if type(layer).__name__ == "Env" for k, v in layer.env_vars}
    assert env[const.DEEPINFER_ACCELERATOR] == "nvidia"


def test_default_image_versions_are_configurable():
    img = default_deepinfer_image(cuda_version="13.1", torch_version="2.14.0+cu133")
    layers = _layers(img)
    pip = [p for layer in layers if type(layer).__name__ == "PipPackages" for p in (layer.packages or ())]
    cmds = "\n".join(c for layer in layers if type(layer).__name__ == "Commands" for c in layer.commands)
    env = {k: v for layer in layers if type(layer).__name__ == "Env" for k, v in layer.env_vars}
    assert "torch==2.14.0+cu133" in pip
    assert "cuda-toolkit-13-1" in cmds
    assert env["CUDA_HOME"] == "/usr/local/cuda-13.1"


def test_source_image_ref_and_repo():
    img = deepinfer_image_from_source(repo="https://git.example/deepinfer.git", ref="v0.1.0")
    cmds = "\n".join(c for layer in _layers(img) if type(layer).__name__ == "Commands" for c in layer.commands)
    assert "--branch v0.1.0 https://git.example/deepinfer.git" in cmds


def test_source_image_extends_default():
    img = deepinfer_image_from_source()
    layers = _layers(img)
    cmd_groups = [getattr(layer, "commands", ()) or () for layer in layers if type(layer).__name__ == "Commands"]
    joined_commands = "\n".join(c for group in cmd_groups for c in group)
    assert "git clone" in joined_commands
    assert "build.sh" in joined_commands
