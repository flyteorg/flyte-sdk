"""Stream a vLLM engine's weights from object storage.

`flyteplugins-vllm` registers a vLLM model loader under
`load_format="flyte-streaming"` through vLLM's `vllm.general_plugins`
entry point, which vLLM runs in every process it starts (the engine core and
each tensor-parallel worker included), so nothing has to be imported or set
in the environment first. `engine_args` returns the arguments that
select it:

```python
from vllm import LLM
from flyteplugins.vllm.model_streamer import engine_args

llm = LLM(**await engine_args("s3://bucket/models/qwen2.5-7b"), max_model_len=4096)
```

The loader is vLLM's default loader with the weights iterator swapped for
`flyte.extras.model_streamer.ModelStreamer`, so vLLM's own
weight loaders still do the placement, sharding and quantization. Each tensor
is handed to them as it arrives and lands in GPU memory without ever being
written to local disk. With tensor parallelism every rank streams the full
checkpoint and keeps its own shard, the same as loading from disk.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import tempfile
import time
import typing

from flyte.extras.model_streamer._streamer import ModelStreamer

LOAD_FORMAT = "flyte-streaming"
"""The vLLM `load_format` that streams weights with `ModelStreamer`."""

# Keys this loader reads out of `model_loader_extra_config`; the rest is
# passed on to vLLM's default loader, which rejects keys it does not know.
_PATH = "flyte_remote_path"
_CHUNK_SIZE = "flyte_chunk_size"
_MAX_CONCURRENCY = "flyte_max_concurrency"
_KEYS = (_PATH, _CHUNK_SIZE, _MAX_CONCURRENCY)


async def engine_args(
    path: str,
    *,
    local_dir: str | pathlib.Path | None = None,
    chunk_size: int | None = None,
    max_concurrency: int | None = None,
) -> dict[str, typing.Any]:
    """Download the model's metadata and return the `LLM` / `EngineArgs` kwargs.

    The config and tokenizer files under `path` are downloaded to
    `local_dir` (a fresh temporary directory by default), which becomes
    `model`; the weights are streamed when the engine loads. The result sets
    `model`, `load_format` and `model_loader_extra_config`; pass any other
    engine argument alongside it.

    Also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` unless it is already set.
    This call reads object storage through obstore, whose runtime does not
    survive `fork()`, and vLLM forks its engine-core process by default; a
    forked engine would hang on its first weight read.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    streamer = ModelStreamer(path, chunk_size=chunk_size, max_concurrency=max_concurrency)
    local = await streamer.download_metadata(local_dir or tempfile.mkdtemp(prefix="model-"))
    return {
        "model": str(local),
        "load_format": LOAD_FORMAT,
        "model_loader_extra_config": {
            _PATH: streamer.path,
            _CHUNK_SIZE: streamer.chunk_size,
            _MAX_CONCURRENCY: streamer.max_concurrency,
        },
    }


def register() -> None:
    """Register the `flyte-streaming` load format with vLLM.

    Called by vLLM itself through the `vllm.general_plugins` entry point, in
    every process it starts. Safe to call more than once; harmless without a
    GPU. Does nothing when vLLM is not installed.
    """
    try:
        from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER, register_model_loader
    except ImportError:
        return
    if LOAD_FORMAT not in _LOAD_FORMAT_TO_MODEL_LOADER:
        register_model_loader(LOAD_FORMAT)(_loader_class())


def _loader_class():
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    class FlyteStreamingModelLoader(DefaultModelLoader):
        """vLLM's default loader, reading the primary weights from a ModelStreamer."""

        def __init__(self, load_config):
            extra = dict(load_config.model_loader_extra_config or {})
            options = {key: extra.pop(key) for key in _KEYS if key in extra}
            if _PATH not in options:
                raise ValueError(
                    f"load_format={LOAD_FORMAT!r} needs model_loader_extra_config[{_PATH!r}]; "
                    "build the engine arguments with flyteplugins.vllm.model_streamer.engine_args()"
                )
            # Secondary weight sources (rare) still load the default way.
            super().__init__(dataclasses.replace(load_config, load_format="auto", model_loader_extra_config=extra))
            self._streamer = ModelStreamer(
                options[_PATH],
                chunk_size=options.get(_CHUNK_SIZE),
                max_concurrency=options.get(_MAX_CONCURRENCY),
            )

        def download_model(self, model_config) -> None:
            # Nothing to download ahead of time: engine_args() fetched the
            # metadata and the weights are streamed in get_all_weights.
            pass

        def get_all_weights(self, model_config, model):
            # The default loader starts this clock in _get_weights_iterator, which
            # the primary weights bypass; without it vLLM logs time since process start.
            self.counter_before_loading_weights = time.perf_counter()
            # vLLM calls this synchronously from inside the engine, possibly on
            # a thread with a running event loop; stream_sync() handles both.
            yield from self._streamer.stream_sync()
            for source in getattr(model, "secondary_weights", ()):
                yield from self._get_weights_iterator(source)

    return FlyteStreamingModelLoader
