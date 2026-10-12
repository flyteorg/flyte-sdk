"""Offline checks for the GPU sharing examples in examples/accelerators.

The driver task builds the child task specs in its own container, where the module is imported
again with only the env vars the driver environment sets. These tests import each example the way
the launcher does, then again with just the driver's env vars, and check that the GPU task keeps
the node selector and the tolerations.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "accelerators"
VARS = ("GPU_NODE_SELECTOR", "GPU_TOLERATIONS")

CASES = [
    ("timeslice_check.py", "ts_env", "gpu_share"),
    ("mig_slice_check.py", "mig_env", "check_mig_slice"),
]


def _import(file_name: str):
    name = f"_accelerator_example_{Path(file_name).stem}"
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _scheduling(task):
    spec = task.pod_template.pod_spec
    return spec.node_selector, [(t.key, t.operator, t.effect) for t in spec.tolerations or []]


@pytest.mark.parametrize("file_name,gpu_env_name,gpu_task_name", CASES)
def test_driver_container_rebuilds_the_same_pod_template(monkeypatch, file_name, gpu_env_name, gpu_task_name):
    monkeypatch.setenv("GPU_NODE_SELECTOR", "pool=gpu-pool")
    monkeypatch.setenv("GPU_TOLERATIONS", "taint-a, taint-b")
    local = _import(file_name)

    expected_env = {"GPU_NODE_SELECTOR": "pool=gpu-pool", "GPU_TOLERATIONS": "taint-a, taint-b"}
    assert local.driver_env.env_vars == expected_env
    assert getattr(local, gpu_env_name).env_vars == expected_env
    assert local.main.env_vars == expected_env

    expected = (
        {"pool": "gpu-pool"},
        [("taint-a", "Exists", "NoSchedule"), ("taint-b", "Exists", "NoSchedule")],
    )
    assert _scheduling(getattr(local, gpu_task_name)) == expected

    # The driver container only has the env vars of the driver environment.
    for var in VARS:
        monkeypatch.delenv(var)
    for k, v in local.driver_env.env_vars.items():
        monkeypatch.setenv(k, v)
    in_driver = _import(file_name)
    assert _scheduling(getattr(in_driver, gpu_task_name)) == expected


@pytest.mark.parametrize("file_name,gpu_env_name,gpu_task_name", CASES)
def test_no_vars_means_no_extra_env(monkeypatch, file_name, gpu_env_name, gpu_task_name):
    for var in VARS:
        monkeypatch.delenv(var, raising=False)
    module = _import(file_name)
    assert module.driver_env.env_vars is None
    assert getattr(module, gpu_env_name).env_vars is None


@pytest.mark.parametrize("file_name", [c[0] for c in CASES])
@pytest.mark.parametrize("only_var", VARS)
def test_one_var_without_the_other_fails_fast(monkeypatch, file_name, only_var):
    for var in VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(only_var, "pool=gpu-pool")
    with pytest.raises(ValueError, match="GPU_NODE_SELECTOR and GPU_TOLERATIONS"):
        _import(file_name)
