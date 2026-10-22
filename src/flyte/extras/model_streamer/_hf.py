"""Hugging Face `transformers` models streamed straight onto a device."""

from __future__ import annotations

import pathlib
import tempfile
import typing

from flyte._logging import logger
from flyte.extras.model_streamer._streamer import (
    LoadResult,
    ModelStreamer,
    _named_tensors,
    _raise_if_incomplete,
    empty_weights,
)

if typing.TYPE_CHECKING:
    import torch
    from torch import nn


async def load_hf_model(
    path: str,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype | None = None,
    model_class: typing.Any = None,
    local_dir: str | pathlib.Path | None = None,
    trust_remote_code: bool = False,
    chunk_size: int | None = None,
    max_concurrency: int | None = None,
) -> tuple[nn.Module, pathlib.Path]:
    """Build a `transformers` model and stream its weights onto `device`.

    The config, tokenizer and generation config are downloaded to
    `local_dir` (a fresh temporary directory by default). The model is built
    with its parameters on `meta`, and each weight is then copied onto
    `device` as soon as it finishes downloading. Weights never touch local
    disk, and host memory holds only the tensors in flight.

    `model_class` defaults to `AutoModelForCausalLM`; any `Auto*` class
    or concrete `PreTrainedModel` subclass works. `dtype` casts
    floating-point weights (`None` keeps the checkpoint's dtype).

    Returns the model in eval mode and the local directory, from which the
    tokenizer can be loaded with `AutoTokenizer.from_pretrained(local_dir)`.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    streamer = ModelStreamer(path, chunk_size=chunk_size, max_concurrency=max_concurrency)
    local = await streamer.download_metadata(local_dir or tempfile.mkdtemp(prefix="model-"))

    config = AutoConfig.from_pretrained(local, trust_remote_code=trust_remote_code)
    cls = model_class or AutoModelForCausalLM
    with empty_weights():
        if hasattr(cls, "from_config"):  # the Auto* factories
            model = cls.from_config(config, trust_remote_code=trust_remote_code)
        else:
            model = cls(config)

    result = await streamer.load_into(
        model, device=device, dtype=dtype, strict=False, key_mapping=_prefix_mapping(model)
    )
    # Like from_pretrained: checkpoint keys the model has no use for are only
    # reported, a parameter left unfilled is an error.
    if result.unexpected_keys:
        logger.warning(f"Ignored {len(result.unexpected_keys)} checkpoint tensors: {result.unexpected_keys[:10]}")
    _raise_if_incomplete(LoadResult(missing_keys=result.missing_keys, unexpected_keys=[]))
    # Buffers were built on the host (see empty_weights); bring them along.
    model.to(device)

    if (local / "generation_config.json").exists() and getattr(model, "generation_config", None) is not None:
        model.generation_config = GenerationConfig.from_pretrained(local)
    return model.eval(), local


def _prefix_mapping(model) -> typing.Callable[[str], str | None]:
    """Reconcile checkpoints saved with and without the base model's prefix.

    A checkpoint saved from `BertForMaskedLM` names its weights
    `bert.encoder...` and one saved from `BertModel` names them
    `encoder...`; `from_pretrained` loads either into either, so this does
    the same by adding or stripping `model.base_model_prefix`.
    """
    names = _named_tensors(model).keys()
    prefix = getattr(model, "base_model_prefix", "") or ""

    def mapping(key: str) -> str | None:
        if key in names or not prefix:
            return key
        if f"{prefix}.{key}" in names:
            return f"{prefix}.{key}"
        if key.startswith(f"{prefix}.") and key[len(prefix) + 1 :] in names:
            return key[len(prefix) + 1 :]
        return key

    return mapping
