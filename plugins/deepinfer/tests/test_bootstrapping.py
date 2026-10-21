"""Unit tests for the DeepInfer bootstrap logic."""

import os
import sys

import pytest

from flyteplugins.deepinfer import _constants as const
from flyteplugins.deepinfer import (
    accelerator_kind,
    create_llm,
    get_actual_num_gpus,
    resolve_model_path,
    setup_common_env,
    setup_cuda_visible_devices,
)

# ---------------------------------------------------------------------------
# resolve_model_path
# ---------------------------------------------------------------------------


def test_resolve_model_path_local_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    out = resolve_model_path(str(tmp_path))
    assert out == str(tmp_path.resolve())
    assert os.path.isdir(out)


def test_resolve_model_path_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_path(str(tmp_path))


def test_resolve_model_path_default_model_env_override(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv(const.DEEPINFER_HF_MODEL_DIR, str(tmp_path))
    assert resolve_model_path(const.DEFAULT_MODEL) == str(tmp_path.resolve())


# ---------------------------------------------------------------------------
# setup_common_env
# ---------------------------------------------------------------------------


def test_setup_common_env_defaults(monkeypatch):
    for key in (
        const.DEEPINFER_USE_NEXTN_DECODE,
        const.DEEPINFER_SAMPLER_FORCE_TOPK,
        const.DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS,
        const.DEEPINFER_EXPERT_COUNT_WINDOW_SIZE,
        const.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL,
        const.DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE,
    ):
        monkeypatch.delenv(key, raising=False)
    setup_common_env()
    assert os.environ[const.DEEPINFER_USE_NEXTN_DECODE] == "5"
    assert os.environ[const.DEEPINFER_SAMPLER_FORCE_TOPK] == "4096"
    assert os.environ[const.DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS] == "32"
    assert os.environ[const.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL] == "30"
    assert os.environ[const.DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE] == "1"


def test_setup_common_env_does_not_overwrite(monkeypatch):
    monkeypatch.setenv(const.DEEPINFER_USE_NEXTN_DECODE, "0")
    setup_common_env()
    assert os.environ[const.DEEPINFER_USE_NEXTN_DECODE] == "0"


def test_setup_common_env_disabled(monkeypatch):
    monkeypatch.delenv(const.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL, raising=False)
    setup_common_env(enable_eplb=False)
    assert const.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL not in os.environ


# ---------------------------------------------------------------------------
# setup_cuda_visible_devices
# ---------------------------------------------------------------------------


def test_setup_cuda_visible_devices_single(monkeypatch):
    monkeypatch.delenv(const.CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.delenv(const.ASCEND_RT_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr("flyteplugins.deepinfer._bootstrapping.accelerator_kind", lambda: "nvidia")
    setup_cuda_visible_devices(engine_rank=0, cp_size=1, device_count=8)
    assert os.environ[const.CUDA_VISIBLE_DEVICES] == "0"


def test_setup_cuda_visible_devices_rank_two(monkeypatch):
    monkeypatch.delenv(const.CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.delenv(const.ASCEND_RT_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr("flyteplugins.deepinfer._bootstrapping.accelerator_kind", lambda: "nvidia")
    setup_cuda_visible_devices(engine_rank=2, cp_size=1, device_count=8)
    assert os.environ[const.CUDA_VISIBLE_DEVICES] == "2"


def test_setup_cuda_visible_devices_cp_two(monkeypatch):
    monkeypatch.delenv(const.CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.delenv(const.ASCEND_RT_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr("flyteplugins.deepinfer._bootstrapping.accelerator_kind", lambda: "nvidia")
    setup_cuda_visible_devices(engine_rank=1, cp_size=2, device_count=8)
    assert os.environ[const.CUDA_VISIBLE_DEVICES] == "2,3"


def test_setup_cuda_visible_devices_ascend(monkeypatch):
    monkeypatch.delenv(const.CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.delenv(const.ASCEND_RT_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr("flyteplugins.deepinfer._bootstrapping.accelerator_kind", lambda: "ascend")
    setup_cuda_visible_devices(engine_rank=0, cp_size=1, device_count=4)
    assert os.environ[const.CUDA_VISIBLE_DEVICES] == "0"
    assert os.environ[const.ASCEND_RT_VISIBLE_DEVICES] == "0"


# ---------------------------------------------------------------------------
# create_llm
# ---------------------------------------------------------------------------


def test_create_llm_passes_kwargs_and_lazy_imports(tmp_path, monkeypatch):
    """create_llm resolves the model, pins devices, and forwards kwargs to LLM."""
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.delenv(const.CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr("flyteplugins.deepinfer._bootstrapping.get_actual_num_gpus", lambda: 8)
    call = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            call.update(kwargs)

    import types

    # Simulate deepinfer not being installed yet: we inject a fake via sys.modules so
    # the lazy import inside create_llm resolves without a real DeepInfer install.
    deepinfer_pkg = types.ModuleType("deepinfer")
    deepinfer_pkg.__path__ = []
    entry_mod = types.ModuleType("deepinfer.entrypoints")
    entry_mod.__path__ = []
    llm_mod = types.ModuleType("deepinfer.entrypoints.llm")
    llm_mod.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "deepinfer", deepinfer_pkg)
    monkeypatch.setitem(sys.modules, "deepinfer.entrypoints", entry_mod)
    monkeypatch.setitem(sys.modules, "deepinfer.entrypoints.llm", llm_mod)

    create_llm(str(tmp_path), engine_rank=0, ep_size=4, cp_size=1, max_model_len=4096)

    assert call["expert_parallel_size"] == 4
    assert call["context_parallel_size"] == 1
    assert call["engine_rank"] == 0
    assert call["max_model_len"] == 4096
    assert call["trust_remote_code"] is True
    assert call["chunked_prefill"] is True
    assert call["decode_sequence"] is False
    assert os.path.realpath(call["model"]) == str(tmp_path.resolve())
    # The engine rank pin ran before import.
    assert const.CUDA_VISIBLE_DEVICES in os.environ


def test_create_llm_override_defaults(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.delenv(const.CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr("flyteplugins.deepinfer._bootstrapping.get_actual_num_gpus", lambda: 8)
    call = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            call.update(kwargs)

    import types

    deepinfer_pkg = types.ModuleType("deepinfer")
    deepinfer_pkg.__path__ = []
    entry_mod = types.ModuleType("deepinfer.entrypoints")
    entry_mod.__path__ = []
    llm_mod = types.ModuleType("deepinfer.entrypoints.llm")
    llm_mod.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "deepinfer", deepinfer_pkg)
    monkeypatch.setitem(sys.modules, "deepinfer.entrypoints", entry_mod)
    monkeypatch.setitem(sys.modules, "deepinfer.entrypoints.llm", llm_mod)

    create_llm(str(tmp_path), engine_rank=0, ep_size=2, trust_remote_code=False)
    assert call["trust_remote_code"] is False


def test_public_helpers_importable_without_deepinfer():
    # Importing the plugin public surface must not require deepinfer/torch.
    assert callable(accelerator_kind)
    assert callable(get_actual_num_gpus)
