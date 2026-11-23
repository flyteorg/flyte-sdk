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
large model, pass `node_cache_dir` to `engine_args`: the first rank on each
node downloads the checkpoint once into that directory, and every rank on the
node reads it from there.
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
_NODE_CACHE_DIR = "flyte_node_cache_dir"
_KEYS = (_PATH, _CHUNK_SIZE, _MAX_CONCURRENCY, _NODE_CACHE_DIR)

# Files the node cache keeps next to the weights: a lock the ranks on a node
# take in turn, and a marker naming the checkpoint once it is fully downloaded.
_CACHE_LOCK = ".flyte-weights.lock"
_CACHE_COMPLETE = ".flyte-weights-complete"


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

    With `node_cache_dir`, the weights are not streamed: the first rank on
    each node downloads the checkpoint into that directory and every rank
    loads it from there, so a tensor-parallel node reads the checkpoint once
    instead of once per rank. The directory must hold the whole checkpoint
    (a node-local disk, or a memory-backed volume), and it can be reused by a
    later engine on the same node, which then skips the download.

    Also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` unless it is already set.
    This call reads object storage through obstore, whose runtime does not
    survive `fork()`, and vLLM forks its engine-core process by default; a
    forked engine would hang on its first weight read.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    streamer = ModelStreamer(path, chunk_size=chunk_size, max_concurrency=max_concurrency)
    local = await streamer.download_metadata(local_dir or tempfile.mkdtemp(prefix="model-"))
    extra: dict[str, typing.Any] = {
        _PATH: streamer.path,
        _CHUNK_SIZE: streamer.chunk_size,
        _MAX_CONCURRENCY: streamer.max_concurrency,
    }
    if node_cache_dir is not None:
        extra[_NODE_CACHE_DIR] = str(node_cache_dir)
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
            self._node_cache_dir = options.get(_NODE_CACHE_DIR)

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
            if self._node_cache_dir:
                yield from _weights_from_node_cache(self._streamer, self._node_cache_dir)
            else:
                yield from self._streamer.stream_sync()
            for source in getattr(model, "secondary_weights", ()):
                yield from self._get_weights_iterator(source)

    return FlyteStreamingModelLoader


def _weights_from_node_cache(
    streamer: ModelStreamer, cache_dir: str | pathlib.Path
) -> typing.Iterator[tuple[str, typing.Any]]:
    """Yield the checkpoint's tensors from a per-node copy, downloading it once.

    The ranks on a node take a file lock in turn. The first one finds no
    completion marker, downloads the checkpoint and writes the marker; the
    rest find the marker and go straight to reading. A marker naming a
    different checkpoint is an error, so a reused directory never loads the
    wrong weights. A download that dies leaves no marker, and the next rank
    downloads again.
    """
    import fcntl

    from safetensors import safe_open

    cache = pathlib.Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / _CACHE_COMPLETE
    with open(cache / _CACHE_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if marker.exists():
                held = marker.read_text().strip()
                if held != streamer.path:
                    raise ValueError(
                        f"{cache} already holds the weights of {held}, not {streamer.path}. "
                        "Use a separate node_cache_dir for each checkpoint."
                    )
            else:
                _run_in_thread(lambda: streamer.download(cache))
                marker.write_text(streamer.path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    # The index, when there is one, decides which file each tensor is read
    # from, the same as streaming; otherwise the first copy of a name wins.
    files = sorted(cache.glob("*.safetensors"))
    index = cache / "model.safetensors.index.json"
    owner: dict[str, str] | None = None
    if index.exists():
        import json

        owner = json.loads(index.read_text()).get("weight_map")
    seen: set[str] = set()
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            for name in f.keys():
                if name in seen or (owner is not None and owner.get(name, file.name) != file.name):
                    continue
                seen.add(name)
                yield name, f.get_tensor(name)


def _run_in_thread(make_coro: typing.Callable[[], typing.Awaitable[typing.Any]]) -> typing.Any:
    """Run a coroutine to completion on a fresh event loop in another thread.

    vLLM calls the loader synchronously, possibly from a thread that already
    runs an event loop, where `asyncio.run` would refuse to start.
    """
    import asyncio
    import threading

    result: dict[str, typing.Any] = {}

    def _run():
        try:
            result["value"] = asyncio.run(make_coro())
        except BaseException as exc:  # re-raised on the calling thread
            result["error"] = exc

    thread = threading.Thread(target=_run, name="model-streamer-download", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
