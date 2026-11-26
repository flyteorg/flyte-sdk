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
written to local disk.

With tensor parallelism every rank streams the full checkpoint and keeps its
own shard, so a node with eight ranks reads the checkpoint eight times. For a
large model, pass `node_cache_dir` to `engine_args`: it downloads the
checkpoint into that directory once, before the engine starts, and vLLM loads
it from there with its own loader.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import pathlib
import shutil
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

# Files the node cache keeps next to the checkpoint: a lock that engines on the
# same node take in turn, and a marker naming the checkpoint once it is fully
# downloaded.
_CACHE_LOCK = ".flyte-weights.lock"
_CACHE_COMPLETE = ".flyte-weights-complete"

# Default reads for the node cache download. Throughput follows the number of reads in
# flight (each one runs at about 20 MB/s from a remote object store), and the reader holds
# about one chunk per read in memory, so many small reads are fast and cheap: 256 x 16 MiB
# is about 4 GiB in flight. Streaming keeps its own, lower defaults.
_NODE_CACHE_CHUNK_SIZE = 16 * 1024 * 1024
_NODE_CACHE_MAX_CONCURRENCY = 256


async def engine_args(
    path: str,
    *,
    local_dir: str | pathlib.Path | None = None,
    chunk_size: int | None = None,
    max_concurrency: int | None = None,
    node_cache_dir: str | pathlib.Path | None = None,
) -> dict[str, typing.Any]:
    """Download the model's metadata and return the `LLM` / `EngineArgs` kwargs.

    The config and tokenizer files under `path` are downloaded to
    `local_dir` (a fresh temporary directory by default), which becomes
    `model`; the weights are streamed when the engine loads. The result sets
    `model`, `load_format` and `model_loader_extra_config`; pass any other
    engine argument alongside it.

    With `node_cache_dir`, the weights are not streamed. The whole checkpoint
    is downloaded into that directory, which becomes `model`, and vLLM loads
    it with its own loader, so a tensor-parallel engine reads the checkpoint
    from object storage once instead of once per rank. `local_dir` is not
    used. The directory must hold the whole checkpoint; a memory-backed
    volume or a fast local disk keeps the load quick. Engines on the same
    node can share it: the first downloads and the rest wait, and a later
    engine, or a container restarted in the same pod, finds the copy and
    skips the download. Every rank of the engine must see the directory, so
    this is for engines on a single node. The download defaults to 256 reads
    of 16 MiB in flight (about 4 GiB of memory); `chunk_size` and
    `max_concurrency` override them.

    Also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` unless it is already set.
    This call reads object storage through obstore, whose runtime does not
    survive `fork()`, and vLLM forks its engine-core process by default; a
    forked engine would hang on its first weight read.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    streamer = ModelStreamer(path, chunk_size=chunk_size, max_concurrency=max_concurrency)
    if node_cache_dir is not None:
        cache = await _download_to_node_cache(
            streamer.path,
            pathlib.Path(node_cache_dir),
            chunk_size=chunk_size or _NODE_CACHE_CHUNK_SIZE,
            max_concurrency=max_concurrency or _NODE_CACHE_MAX_CONCURRENCY,
        )
        return {"model": str(cache), "load_format": "auto"}
    local = await streamer.download_metadata(local_dir or tempfile.mkdtemp(prefix="model-"))
    extra: dict[str, typing.Any] = {
        _PATH: streamer.path,
        _CHUNK_SIZE: streamer.chunk_size,
        _MAX_CONCURRENCY: streamer.max_concurrency,
    }
    return {"model": str(local), "load_format": LOAD_FORMAT, "model_loader_extra_config": extra}


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


async def _download_to_node_cache(
    path: str, cache: pathlib.Path, *, chunk_size: int, max_concurrency: int
) -> pathlib.Path:
    """Download the checkpoint into `cache` unless a complete copy is already there.

    Engines on the same node take a file lock in turn. The first finds no
    completion marker, clears whatever an unfinished download left behind
    and downloads; the rest find the marker and return at once. A marker
    naming a different checkpoint is an error, so a reused directory never
    serves the wrong weights.
    """
    import flyte.storage as storage

    lock = await asyncio.to_thread(_lock_node_cache, cache, path)
    if lock is None:
        return cache
    try:
        await storage.get(path, str(cache), recursive=True, chunk_size=chunk_size, max_concurrency=max_concurrency)
        await asyncio.to_thread((cache / _CACHE_COMPLETE).write_text, path)
    finally:
        _unlock(lock)
    return cache


def _lock_node_cache(cache: pathlib.Path, path: str) -> typing.IO | None:
    """Take the node cache's lock and return it, or return None if the copy is complete.

    Before returning the lock, clears what an unfinished download left behind.
    """
    import fcntl

    cache.mkdir(parents=True, exist_ok=True)
    lock = open(cache / _CACHE_LOCK, "a")  # released by _unlock
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        marker = cache / _CACHE_COMPLETE
        if marker.exists():
            held = marker.read_text().strip()
            if held != path:
                raise ValueError(
                    f"{cache} already holds the weights of {held}, not {path}. "
                    "Use a separate node_cache_dir for each checkpoint."
                )
            _unlock(lock)
            return None
        _clear_partial_download(cache)
    except BaseException:
        _unlock(lock)
        raise
    return lock


def _unlock(lock: typing.IO) -> None:
    import fcntl

    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


def _clear_partial_download(cache: pathlib.Path) -> None:
    """Remove what a download that died partway left in `cache`.

    That is weight shards and index files, which vLLM would otherwise load
    with the new checkpoint, and the reader's `.flyte-download-*` staging
    folders, which on a memory-backed volume hold pod memory until the pod ends.
    """
    for path in cache.iterdir():
        if path.name.startswith(".flyte-download-") and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.name.endswith((".safetensors", ".safetensors.index.json")):
            path.unlink()
