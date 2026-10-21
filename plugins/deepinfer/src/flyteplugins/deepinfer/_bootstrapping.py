"""Bootstrap a DeepInfer engine inside a Flyte task.

DeepInfer is unusual in *when* its environment must be configured: the CUDA /
CANN visible-device list and several runtime flags only take effect if they are
set *before* `import torch` / `import deepinfer` runs (importing deepinfer
initialises the platform and pins the visible devices; a CUDA context created
earlier makes a later `CUDA_VISIBLE_DEVICES` assignment silently ineffective).
This module centralises that ordering so the body of a Flyte task is a straight
call instead of a fragile prologue:

```python
import flyte
from flyteplugins.deepinfer import create_llm

env = flyte.TaskEnvironment(name="deepinfer-batch", image="...")

@env.task(resources=flyte.Resources(cpu="48", memory="512Gi", gpu="SM100:8"))
async def score(prompts: list[str], model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017") -> list[str]:
    llm = create_llm(model, engine_rank=0, ep_size=8)
    outputs = llm.generate(prompts, sampling_params=..., use_tqdm=False)
    return [out[0].text for out in outputs]
```

Every function here imports upstream DeepInfer lazily — nothing is imported at
module load, so a task graph compiles (and unit tests run) on machines without
DeepInfer, a GPU, or even CUDA.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from flyteplugins.deepinfer import _constants as const

if TYPE_CHECKING:
    from deepinfer.entrypoints.llm import LLM


def accelerator_kind() -> str:
    """Return `"nvidia"` or `"ascend"` for the current host.

    Uses `deep_cluster.hardware.accelerator_kind` (NVML / CANN probing). Falls
    back to the `DEEPINFER_ACCELERATOR` env var when `deep_cluster` is not
    installed (e.g. while authoring tasks off-host). One of
    `flyteplugins.deepinfer._constants.ACCELERATOR_*`.
    """
    try:
        from deep_cluster.hardware import accelerator_kind as _kind

        return _kind()
    except Exception:  # pragma: no cover - env/platform dependent
        return os.environ.get(const.DEEPINFER_ACCELERATOR, const.ACCELERATOR_NVIDIA)


def get_actual_num_gpus() -> int:
    """Number of GPUs (or NPUs) visible to this node.

    Lazily imports `deep_cluster.hardware.get_actual_num_gpus`; falls back to
    `torch.cuda.device_count` if unavailable.
    """
    try:
        from deep_cluster.hardware import get_actual_num_gpus as _count

        return _count()
    except Exception:  # pragma: no cover - env/platform dependent
        import torch

        return torch.cuda.device_count()


def setup_cuda_visible_devices(engine_rank: int, cp_size: int, device_count: int) -> None:
    """Set `CUDA_VISIBLE_DEVICES` (and the Ascend equivalent) for an engine rank.

    DeepInfer runs several *engines* (one per data-parallel rank) inside one node,
    each owning `cp_size` devices. This pins the current process to its slice so
    that a spawned engine never grabs a device another rank is using.

    Args:
        engine_rank: The data-parallel engine rank this process will drive.
        cp_size: Context-parallel size (devices per engine).
        device_count: Total devices visible on the node (use `get_actual_num_gpus`).
    """
    device_offset = int(os.environ.get(const.DEVICE_OFFSET, "0"))
    local_engine_rank = engine_rank * cp_size % device_count // cp_size
    indices = range(local_engine_rank * cp_size, (local_engine_rank + 1) * cp_size)
    visible = ",".join(str(device_offset + i) for i in indices)
    os.environ[const.CUDA_VISIBLE_DEVICES] = visible
    if accelerator_kind() == const.ACCELERATOR_ASCEND:
        os.environ[const.ASCEND_RT_VISIBLE_DEVICES] = visible


def setup_common_env(
    *,
    sampler_force_topk: int = 4096,
    set_override_experts: bool = True,
    enable_eplb: bool = True,
    model_name: Optional[str] = None,
) -> None:
    """Configure the DeepInfer runtime environment with defaults a Flyte batch task
    wants, without overwriting anything the user has already set.

    `os.environ.setdefault` is used throughout, so explicit values survive. The
    defaults mirror DeepInfer's own offline-inference example (`examples/_config.py`).

    Args:
        sampler_force_topk: Value for `DEEPINFER_SAMPLER_FORCE_TOPK`.
        set_override_experts: Add `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` (the engine
            only treats it as a lower bound).
        enable_eplb: Turn on expert-parallel load balancing scheduling.
        model_name: Optional model id; accepted for symmetry with upstream but not
            required here.
    """
    os.environ.setdefault(const.DEEPINFER_USE_NEXTN_DECODE, "5")
    os.environ.setdefault(const.DEEPINFER_SAMPLER_FORCE_TOPK, str(sampler_force_topk))
    if set_override_experts:
        os.environ.setdefault(const.DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS, "32")
    os.environ.setdefault(const.DEEPINFER_EXPERT_COUNT_WINDOW_SIZE, "3")
    if enable_eplb:
        os.environ.setdefault(const.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL, "30")
        os.environ.setdefault(const.DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE, "1")


def resolve_model_path(model_name: str | Path) -> str:
    """Resolve a model reference to a local Hugging Face checkpoint directory.

    A path that already contains a `config.json` is used as-is (this covers a local
    checkpoint *or* a Flyte `flyte.io.Dir` / `RunOutput` mounted into the task, whose
    path you can pass through directly). Anything else is treated as an HF model id and
    resolved with `huggingface_hub.snapshot_download`, honouring
    `DEEPINFER_HF_MODEL_DIR` for the default model.

    Args:
        model_name: HF model id, local checkpoint directory, or mounted Flyte dir path.

    Returns:
        The absolute path to a directory containing `config.json`.
    """
    text = os.fspath(model_name)
    if text == const.DEFAULT_MODEL and (override := os.environ.get(const.DEEPINFER_HF_MODEL_DIR)):
        text = override
    path = Path(text).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(text))
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"DeepInfer checkpoint requires config.json: {path}")
    return str(path.resolve())


def create_llm(
    model_name: str | Path,
    engine_rank: int = 0,
    *,
    ep_size: int,
    cp_size: int = 1,
    env: Optional[dict[str, str]] = None,
    **llm_kwargs: Any,
) -> "LLM":
    """Build a DeepInfer `LLM` for offline inference inside a task.

    The construction order matters and is handled here: environment variables are
    applied, the model path is resolved, per-rank visible devices are pinned, and
    *then* `deepinfer` is imported and the engine built. `llm_kwargs` are passed
    straight through to `deepinfer.entrypoints.llm.LLM` and override these sensible
    defaults: `trust_remote_code=True`, `chunked_prefill=True`,
    `decode_sequence=False`, `disable_log_stats=False`.

    Args:
        model_name: HF id / local checkpoint / mounted flyte dir (see `resolve_model_path`).
        engine_rank: Data-parallel engine rank for this process (see `setup_cuda_visible_devices`).
        ep_size: Expert-parallel size (number of devices split across MoE experts).
        cp_size: Context-parallel size (devices per engine). Defaults to 1.
        env: Extra environment variables applied before import.
        **llm_kwargs: Remaining args forwarded to `deepinfer`'s `LLM` (for example
            `gpu_memory_utilization`, `max_model_len`, `max_num_batched_tokens`).

    Returns:
        A constructed `deepinfer.entrypoints.llm.LLM`. Calling its `generate` method
        produces completions for the current rank's engine.
    """
    import os as _os

    if env:
        _os.environ.update(env)
    model_path = resolve_model_path(model_name)
    setup_cuda_visible_devices(engine_rank, cp_size, get_actual_num_gpus())

    # Imported lazily and only now: `import deepinfer` after the visible devices are
    # pinned initializes the platform for the correct GPU set.
    from deepinfer.entrypoints.llm import LLM

    return LLM(
        model=model_path,
        expert_parallel_size=ep_size,
        context_parallel_size=cp_size,
        engine_rank=engine_rank,
        **{
            "trust_remote_code": True,
            "chunked_prefill": True,
            "decode_sequence": False,
            "disable_log_stats": False,
            **llm_kwargs,
        },
    )
