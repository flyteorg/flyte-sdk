"""Unit tests for the llama-cpp-fserve shim (GGUF resolution and argv rewriting)."""

from pathlib import Path

import pytest

from flyteplugins.llamacpp._server import build_command, find_gguf


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


# find_gguf


def test_find_gguf_direct_file(tmp_path):
    gguf = _touch(tmp_path / "model.gguf")
    assert find_gguf(gguf) == gguf


def test_find_gguf_in_directory(tmp_path):
    gguf = _touch(tmp_path / "model-Q4_K_M.gguf")
    assert find_gguf(str(tmp_path)) == gguf


def test_find_gguf_nested(tmp_path):
    gguf = _touch(tmp_path / "sub" / "dir" / "model.gguf")
    assert find_gguf(str(tmp_path)) == gguf


def test_find_gguf_prefers_top_level_over_subdir_draft(tmp_path):
    """A model-dir that also holds the draft/MTP GGUF in a subdirectory (the object-store
    FUSE layout) must resolve to the top-level model, not the nested draft -- even though the
    draft's path sorts first."""
    model = _touch(tmp_path / "Qwen3-27B-Q4_K_M.gguf")
    _touch(tmp_path / "MTP" / "mtp-draft-Q4_0.gguf")
    assert find_gguf(str(tmp_path)) == model


def test_find_gguf_prefers_first_shard(tmp_path):
    # "a-..." sorts before the shard files; the first shard must still win.
    _touch(tmp_path / "a-mmproj.gguf")
    shard1 = _touch(tmp_path / "model-00001-of-00002.gguf")
    _touch(tmp_path / "model-00002-of-00002.gguf")
    assert find_gguf(str(tmp_path)) == shard1


def test_find_gguf_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"No \.gguf files found"):
        find_gguf(str(tmp_path))


# build_command


def test_build_command_resolves_model_dir(tmp_path):
    gguf = _touch(tmp_path / "model.gguf")
    cmd = build_command(["--model-dir", str(tmp_path), "--alias", "m", "--port", "8080"])
    assert cmd[1:] == ["--model", gguf, "--alias", "m", "--port", "8080"]
    assert cmd[0].endswith("llama-server")


def test_build_command_resolves_draft_model_dir(tmp_path):
    target = _touch(tmp_path / "target" / "model.gguf")
    draft = _touch(tmp_path / "draft" / "draft.gguf")
    cmd = build_command(["--model-dir", str(tmp_path / "target"), "--draft-model-dir", str(tmp_path / "draft")])
    assert cmd[1:] == ["--model", target, "--model-draft", draft]


def test_build_command_passes_other_args_through():
    cmd = build_command(["--hf-repo", "org/repo:Q4_K_M", "--alias", "m", "--jinja"])
    assert cmd[1:] == ["--hf-repo", "org/repo:Q4_K_M", "--alias", "m", "--jinja"]


def test_build_command_missing_dir_value_raises():
    with pytest.raises(ValueError, match="--model-dir requires a value"):
        build_command(["--model-dir"])
