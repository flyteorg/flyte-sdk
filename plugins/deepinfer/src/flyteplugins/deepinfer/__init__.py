"""DeepInfer plugin for Flyte.

Run [DeepSeek's DeepInfer](https://github.com/deepseek-ai/DeepInfer) mid/large-scale
offline inference as Flyte tasks. DeepInfer is the inference framework that powers
DeepSeek's production serving (optimised kernels, MoE expert-parallel load balancing,
speculative/confidence-scheduled verification, structured output), derived from vLLM
and supporting DeepSeek-V4.1-Flash on NVIDIA SM100 GPUs and Huawei Ascend 950 NPUs.

This plugin provides the *task-side* plumbing that is easy to get subtly wrong and
hard to debug once the probe is burned:

- `create_llm` — construct a DeepInfer engine with the correct environment ordering
  (visible devices and runtime flags must be set *before* `import deepinfer`).
- `resolve_model_path` — turn an HF model id, a local checkpoint, or a mounted flyte
  dir into a path DeepInfer can load.
- `setup_common_env` / `setup_cuda_visible_devices` — runtime environment defaults and
  per-engine GPU pinning.
- `launch_dp` / `run_engines` — multi-process data-parallel execution over a task's GPUs.
- `sampling_params` / `structured_output` — import-free sampling and constrained
  decoding configuration.
- `default_deepinfer_image` / `deepinfer_image_from_source` — container image helpers.
- `clustered_deepinfer_environment` — a `ClusteredTaskEnvironment` (JobSet) shaped to
  DeepInfer's one-node-launcher-per-pod multi-node model.

DeepInfer is built from source, not published to PyPI, and targets specific
hardware, so nothing here imports DeepInfer at module load; all upstream imports are
lazy inside the functions above.
"""

from flyteplugins.deepinfer._bootstrapping import (
    accelerator_kind,
    create_llm,
    get_actual_num_gpus,
    resolve_model_path,
    setup_common_env,
    setup_cuda_visible_devices,
)
from flyteplugins.deepinfer._clustered import clustered_deepinfer_environment
from flyteplugins.deepinfer._constants import ACCELERATOR_ASCEND, ACCELERATOR_NVIDIA
from flyteplugins.deepinfer._image import deepinfer_image_from_source, default_deepinfer_image
from flyteplugins.deepinfer._launch import launch_dp, run_dp, run_engines
from flyteplugins.deepinfer._sampling import sampling_params, structured_output

__all__ = [
    "ACCELERATOR_ASCEND",
    "ACCELERATOR_NVIDIA",
    "accelerator_kind",
    "clustered_deepinfer_environment",
    "create_llm",
    "deepinfer_image_from_source",
    "default_deepinfer_image",
    "get_actual_num_gpus",
    "launch_dp",
    "resolve_model_path",
    "run_dp",
    "run_engines",
    "sampling_params",
    "setup_common_env",
    "setup_cuda_visible_devices",
    "structured_output",
]
