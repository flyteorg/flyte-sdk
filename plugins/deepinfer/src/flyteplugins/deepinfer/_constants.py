"""Constants for the DeepInfer flyte plugin.

DeepInfer is DeepSeek's high-performance inference framework. This release pins a
single supported model (DeepSeek-V4.1-Flash) and two accelerator targets
(NVIDIA SM100 GPUs, Huawei Ascend 950 NPUs), and must be built from source on a
Python 3.12 environment with a CUDA 13 / CANN 9.2 toolchain and GCC >= 13. The
values here encode those constraints so the plugin's bootstrap and image helpers
stay in sync with the upstream project.
"""

from __future__ import annotations

# The only release-supported model (a DeepSeek-GEMMoE V4.1-Flash checkpoint).
# When unset, ``resolve_model_path`` downloads this from the Hugging Face hub into
# the HF cache on first use; ``DEEPINFER_HF_MODEL_DIR`` redirects it to a local copy.
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

# Fairly arbitrary lower bound on the plugin import surface: DeepInfer itself must be
# built from source, so unlike a PyPI-pin this is informational only. It mirrors the
# upstream project version (0.1.0) and is surfaced in error messages.
DEEPINFER_MIN_VERSION = "0.1.0"

# Environment variables the DeepInfer runtime reads. ``setup_common_env`` layers
# sensible defaults over the ones a full serving stack needs; users may override each
# through ``env=`` / ``env_vars=``.
DEEPINFER_ACCELERATOR = "DEEPINFER_ACCELERATOR"
DEEPINFER_HF_MODEL_DIR = "DEEPINFER_HF_MODEL_DIR"
DEEPINFER_USE_NEXTN_DECODE = "DEEPINFER_USE_NEXTN_DECODE"
DEEPINFER_SAMPLER_FORCE_TOPK = "DEEPINFER_SAMPLER_FORCE_TOPK"
DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS = "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS"
DEEPINFER_EXPERT_COUNT_WINDOW_SIZE = "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE"
DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL = "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL"
DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE = "DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE"
DEEPINFER_BATCH_INVARIANT = "DEEPINFER_BATCH_INVARIANT"
DEEPINFER_ENABLE_MTP_VARLEN = "DEEPINFER_ENABLE_MTP_VARLEN"
DEEPINFER_LOGGING_INTERVAL_STEP = "DEEPINFER_LOGGING_INTERVAL_STEP"

# Accelerator selectors recognized by ``DEEPINFER_ACCELERATOR``.
ACCELERATOR_NVIDIA = "nvidia"
ACCELERATOR_ASCEND = "ascend"

# Multi-process launch helpers read the distributed rank from the environment when the
# task is launched across nodes (mirrors DeepInfer's own examples).
RANK = "RANK"
WORLD_SIZE = "WORLD_SIZE"
DEVICE_OFFSET = "DEVICE_OFFSET"

# Device-visibility env vars set per engine rank before ``import deepinfer``.
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
ASCEND_RT_VISIBLE_DEVICES = "ASCEND_RT_VISIBLE_DEVICES"
