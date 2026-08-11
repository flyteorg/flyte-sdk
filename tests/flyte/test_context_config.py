"""Tests for the task context config file (write side) and flyte.load_context (restore side)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import flyte
from flyte._context import root_context_var
from flyte._context_config import context_config_path, load_context, write_context_config

A0_PARAMS = {
    "inputs": "s3://bucket/run/a0/inputs.pb",
    "outputs_path": "s3://bucket/run/a0/output",
    "version": "v1",
    "run_base_dir": "s3://bucket/run",
    "raw_data_path": "s3://bucket/run/raw",
    "checkpoint_path": "s3://bucket/run/a0/ckpt",
    "prev_checkpoint": "",
    "name": "a0",
    "run_name": "test-run",
    "run_start_time": "2026-08-11T01:02:03Z",
    "project": "proj",
    "domain": "dev",
    "org": "acme",
    "debug": True,
    "interactive_mode": False,
    "image_cache": None,
    "tgz": "s3://bucket/code.tgz",
    "pkl": None,
    "dest": ".",
    "resolver": "flyte._internal.resolvers.default.DefaultTaskResolver",
    "resolver_args": ("mod", "my_mod", "instance", "task1"),
}


def test_write_context_config(tmp_path):
    path = write_context_config(A0_PARAMS, base_dir=tmp_path)

    assert path == tmp_path / ".flyte" / "config.json"
    assert path == context_config_path(tmp_path)
    config = json.loads(path.read_text())
    assert config["config_version"] == 1
    assert config["inputs"] == A0_PARAMS["inputs"]
    assert config["outputs_path"] == A0_PARAMS["outputs_path"]
    assert config["raw_data_path"] == A0_PARAMS["raw_data_path"]
    assert config["run_name"] == "test-run"
    assert config["name"] == "a0"
    assert config["resolver_args"] == ["mod", "my_mod", "instance", "task1"]
    # debug/interactive flags are runtime-mode toggles, not context, and are not persisted
    assert "debug" not in config


def test_load_context_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="No task context config found"):
        load_context()
    with pytest.raises(FileNotFoundError, match="No task context config found"):
        load_context(path=tmp_path / "nope" / "config.json")


def test_load_context_missing_required_fields(tmp_path):
    incomplete = dict(A0_PARAMS)
    incomplete["raw_data_path"] = None
    incomplete["org"] = None
    path = write_context_config(incomplete, base_dir=tmp_path)
    with pytest.raises(ValueError, match="missing required fields") as exc_info:
        load_context(path=path)
    assert "raw_data_path" in str(exc_info.value)
    assert "org" in str(exc_info.value)


def test_load_context_restores_task_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("_F_PATH_REWRITE", raising=False)
    write_context_config(A0_PARAMS)

    prev_ctx = root_context_var.get()
    try:
        with patch("flyte._initialize.init_in_cluster", MagicMock()) as mock_init:
            tctx = load_context()

        mock_init.assert_called_once_with(org="acme", project="proj", domain="dev")
        assert tctx.action.name == "a0"
        assert tctx.action.run_name == "test-run"
        assert tctx.action.project == "proj"
        assert tctx.action.domain == "dev"
        assert tctx.action.org == "acme"
        assert tctx.version == "v1"
        assert tctx.raw_data_path.path == "s3://bucket/run/raw"
        assert tctx.input_path == "s3://bucket/run/a0/inputs.pb"
        assert tctx.output_path == "s3://bucket/run/a0/output"
        assert tctx.run_base_dir == "s3://bucket/run"
        assert tctx.checkpoint_paths.checkpoint_path == "s3://bucket/run/a0/ckpt"
        assert tctx.code_bundle is not None
        assert tctx.code_bundle.tgz == "s3://bucket/code.tgz"
        assert tctx.interactive_mode is True
        assert tctx.mode == "remote"
        assert tctx.run_start_time.isoformat() == "2026-08-11T01:02:03+00:00"

        # The context is installed: flyte.ctx() now returns the restored task context.
        assert flyte.ctx() is tctx
    finally:
        root_context_var.set(prev_ctx)


def test_load_context_resolves_template_names_from_env(tmp_path, monkeypatch):
    params = dict(A0_PARAMS)
    params["run_name"] = "{{.runName}}"
    params["name"] = "{{.actionName}}"
    path = write_context_config(params, base_dir=tmp_path)
    monkeypatch.setenv("RUN_NAME", "env-run")
    monkeypatch.setenv("ACTION_NAME", "env-action")

    prev_ctx = root_context_var.get()
    try:
        with patch("flyte._initialize.init_in_cluster", MagicMock()):
            tctx = load_context(path=path)
        assert tctx.action.run_name == "env-run"
        assert tctx.action.name == "env-action"
    finally:
        root_context_var.set(prev_ctx)


def test_prepare_launch_json_writes_context_config(tmp_path, monkeypatch):
    from flyte._debug.vscode import prepare_launch_json

    monkeypatch.chdir(tmp_path)
    ctx = SimpleNamespace(params=dict(A0_PARAMS))
    prepare_launch_json(ctx, pid=1234)

    assert (tmp_path / ".vscode" / "launch.json").is_file()
    config_path = tmp_path / ".flyte" / "config.json"
    assert config_path.is_file()
    config = json.loads(config_path.read_text())
    assert config["run_name"] == "test-run"
    assert config["raw_data_path"] == A0_PARAMS["raw_data_path"]
