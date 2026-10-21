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


def test_source_image_extends_default():
    img = deepinfer_image_from_source()
    layers = _layers(img)
    cmd_groups = [getattr(layer, "commands", ()) or () for layer in layers if type(layer).__name__ == "Commands"]
    joined_commands = "\n".join(c for group in cmd_groups for c in group)
    assert "git clone" in joined_commands
    assert "build.sh" in joined_commands
