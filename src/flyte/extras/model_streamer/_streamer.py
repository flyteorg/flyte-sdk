"""Public face of the safetensors streamer behind the vLLM / SGLang apps.

`SafeTensorsStreamer` (`flyte.app.extras._model_loader`) is what the app
shims use, driven by env vars read at import. Its one public method,
`get_tensors()`, drives an `asyncio.Runner` and so fails inside a running
event loop, which is where every async task body runs. `ModelStreamer`
exposes the same streaming as an async iterator, a thread-backed sync iterator
that works from any thread, and helpers that place each tensor on its device as
it arrives, so a model never has to sit in host memory or on local disk in
full.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import pathlib
import queue
import threading
import time
import typing

from flyte._logging import logger

if typing.TYPE_CHECKING:
    import torch
    from torch import nn

# Files whose names match these are weights; everything else under a model
# prefix (config, tokenizer, generation config, chat template) is metadata.
SAFETENSORS_PATTERN = "*.safetensors"

_END = object()

# obstore drives a tokio runtime that does not survive fork(): once a process
# has used it, every obstore call in a child forked from that process blocks
# forever, with no error. Remember which process streamed so a forked child
# fails fast instead. (Other obstore users in the parent, flyte's own I/O
# included, taint a fork the same way; this only catches our own.)
_obstore_owner: dict[str, int] = {}


def _claim_obstore() -> None:
    owner = _obstore_owner.setdefault("pid", os.getpid())
    if owner != os.getpid():
        raise RuntimeError(
            f"ModelStreamer was used in process {owner}, and this process ({os.getpid()}) was forked "
            "from it. obstore's runtime does not survive fork(), so reading object storage here would hang. "
            "Start the child with the 'spawn' method instead (for vLLM, set "
            "VLLM_WORKER_MULTIPROC_METHOD=spawn, which flyteplugins.vllm.model_streamer.engine_args() does)."
        )


def _default_chunk_size() -> int:
    from flyte.app.extras._model_loader.config import CHUNK_SIZE

    return CHUNK_SIZE


def _default_max_concurrency() -> int:
    from flyte.app.extras._model_loader.config import MAX_CONCURRENCY

    return MAX_CONCURRENCY


@dataclasses.dataclass
class LoadResult:
    """What `ModelStreamer.load_into` did not match."""

    missing_keys: list[str]
    """Parameters of the module that no streamed tensor filled."""

    unexpected_keys: list[str]
    """Streamed tensors with no matching parameter or buffer in the module."""


class ModelStreamer:
    """Stream the safetensors weights under a remote prefix, tensor by tensor.

    `path` is the object-store prefix of a Hugging Face style model directory
    (`s3://bucket/models/qwen2.5-7b`) holding `*.safetensors` files and,
    optionally, a `model.safetensors.index.json`. When the index is present it
    decides which file each tensor is read from; otherwise every
    `*.safetensors` file directly under the prefix is read and the first copy
    of a duplicated tensor name wins.

    Each tensor is fetched as `chunk_size` byte ranges, `max_concurrency`
    at a time, and yielded as soon as its last range lands, so loading overlaps
    the download instead of waiting for it. Peak host memory is roughly the
    tensors in flight, not the model.

    Credentials come from the task's storage configuration (or the ambient
    cloud environment outside a task), the same way `flyte.io.File` reads.
    """

    def __init__(
        self,
        path: str,
        *,
        chunk_size: int | None = None,
        max_concurrency: int | None = None,
    ):
        self.path = path.rstrip("/")
        self.chunk_size = chunk_size or _default_chunk_size()
        self.max_concurrency = max_concurrency or _default_max_concurrency()

    def _streamer(self):
        from flyte.app.extras._model_loader.loader import SafeTensorsStreamer

        # local_path is unused by the streamer (weights never touch disk), but
        # the constructor still requires it.
        return SafeTensorsStreamer(
            self.path,
            "/dev/null",
            chunk_size=self.chunk_size,
            max_concurrency=self.max_concurrency,
        )

    async def stream(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> typing.AsyncIterator[tuple[str, torch.Tensor]]:
        """Yield `(name, tensor)` pairs in completion order, not file order.

        With `device` set, each tensor is copied there as it arrives and its
        host buffer is released; with `dtype` set, floating-point tensors are
        cast (integer and boolean tensors are left alone).

        Raises `RuntimeError` in a process forked from one that already
        streamed: object-storage reads cannot work across `fork()`.
        """
        _claim_obstore()
        start = time.perf_counter()
        count = 0
        nbytes = 0
        # get_tensors() wraps this in an asyncio.Runner, which cannot run
        # inside an event loop.
        async for name, tensor in self._streamer()._get_tensors_async():
            count += 1
            nbytes += tensor.numel() * tensor.element_size()
            yield name, _place(tensor, device, dtype)
        elapsed = time.perf_counter() - start
        logger.info(
            f"Streamed {count} tensors ({nbytes / 2**30:.2f} GiB) from {self.path} "
            f"in {elapsed:.2f}s ({nbytes / 2**30 / max(elapsed, 1e-9):.2f} GiB/s)"
        )

    def stream_sync(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> typing.Iterator[tuple[str, torch.Tensor]]:
        """Synchronous `stream`, safe to call from any thread.

        The download runs on its own event loop in a background thread, so this
        works both from plain synchronous code and from a thread that already
        has a loop running (where flyte's own `get_tensors()` raises). The
        hand-off queue is bounded: a slow consumer pauses the download rather
        than letting finished tensors pile up in host memory.
        """
        q: queue.Queue = queue.Queue(maxsize=max(self.max_concurrency, 1))
        stop = threading.Event()

        def _put(item) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        async def _pump():
            # Device placement happens on the consumer thread, which owns the
            # CUDA context the caller set up.
            async for item in self.stream():
                if not await asyncio.to_thread(_put, item):
                    return

        def _run():
            try:
                asyncio.run(_pump())
            except BaseException as exc:  # handed to the consumer, re-raised there
                _put(exc)
            else:
                _put(_END)

        thread = threading.Thread(target=_run, name="model-streamer", daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _END:
                    return
                if isinstance(item, BaseException):
                    raise item
                name, tensor = item
                yield name, _place(tensor, device, dtype)
        finally:
            stop.set()
            thread.join()

    async def state_dict(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Collect every tensor into a dict, each placed on `device` as it lands."""
        return {name: tensor async for name, tensor in self.stream(device=device, dtype=dtype)}

    async def load_into(
        self,
        module: nn.Module,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        strict: bool = True,
        key_mapping: typing.Callable[[str], str | None] | None = None,
    ) -> LoadResult:
        """Stream the weights straight into `module`'s parameters and buffers.

        Each tensor replaces the module's entry as it arrives (`assign`
        semantics, as in `load_state_dict(assign=True)`), so the module can be
        built on the `meta` device with `empty_weights` and never
        allocate its weights twice. Tensors go to `device` when given,
        otherwise to the device of the entry they replace (`meta` entries
        need an explicit `device`).

        `key_mapping` renames a checkpoint key to a module key, or returns
        `None` to skip it. Checkpoints usually store a tied weight once (an
        embedding shared with the output head), so when the module has a
        `tie_weights()` method, as Hugging Face models do, it is called once
        the stream ends. With `strict` (the default), any parameter still on
        `meta` after that, or any checkpoint key left unmatched, raises
        `KeyError`.
        """
        import torch

        targets = _named_tensors(module)
        unexpected: list[str] = []
        async for name, tensor in self.stream():
            key = key_mapping(name) if key_mapping is not None else name
            if key is None:
                continue
            if key not in targets:
                unexpected.append(name)
                continue
            owner, attr, is_param = targets[key]
            current = owner._parameters[attr] if is_param else owner._buffers[attr]
            target_device = device
            if target_device is None:
                if current is None or current.device.type == "meta":
                    raise ValueError(f"{key!r} is on the meta device; pass device= to place it")
                target_device = current.device
            if current is not None and tuple(current.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"{key!r}: checkpoint shape {tuple(tensor.shape)} != module shape {tuple(current.shape)}"
                )
            placed = _place(tensor, target_device, dtype)
            if is_param:
                requires_grad = current.requires_grad if current is not None else False
                owner._parameters[attr] = torch.nn.Parameter(placed, requires_grad=requires_grad)
            else:
                owner._buffers[attr] = placed

        tie_weights = getattr(module, "tie_weights", None)
        if callable(tie_weights):
            tie_weights()

        result = LoadResult(missing_keys=missing_parameters(module), unexpected_keys=unexpected)
        if strict:
            _raise_if_incomplete(result)
        return result

    async def download_metadata(self, local_dir: str | pathlib.Path) -> pathlib.Path:
        """Download everything under the prefix except the weights.

        Config, tokenizer and generation-config files are small and the
        libraries that read them want a local directory; the `*.safetensors`
        files are skipped because `stream` reads them directly.
        """
        from flyte.storage._storage import _get_obstore_bypass

        _claim_obstore()
        local = pathlib.Path(local_dir)
        await asyncio.to_thread(local.mkdir, parents=True, exist_ok=True)
        await _get_obstore_bypass(self.path, str(local), recursive=True, exclude=[SAFETENSORS_PATTERN])
        return local


def _place(tensor: torch.Tensor, device, dtype) -> torch.Tensor:
    if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
        return tensor.to(device=device, dtype=dtype)
    if device is not None:
        return tensor.to(device=device)
    return tensor


def _named_tensors(module: nn.Module) -> dict[str, tuple[nn.Module, str, bool]]:
    """Map every parameter and persistent-or-not buffer name to its owner.

    Tied parameters appear under each name that reaches them; filling one name
    replaces only that owner's entry, which is why tied models must re-tie
    after loading.
    """
    out: dict[str, tuple[nn.Module, str, bool]] = {}
    for prefix, sub in module.named_modules(remove_duplicate=False):
        dot = f"{prefix}." if prefix else ""
        for attr in sub._parameters:
            out[dot + attr] = (sub, attr, True)
        for attr in sub._buffers:
            out[dot + attr] = (sub, attr, False)
    return out


def missing_parameters(module: nn.Module) -> list[str]:
    """Parameters and buffers still on the `meta` device."""
    return [
        name for name, tensor in [*module.named_parameters(), *module.named_buffers()] if tensor.device.type == "meta"
    ]


def _raise_if_incomplete(result: LoadResult) -> None:
    problems = []
    if result.missing_keys:
        problems.append(f"missing {len(result.missing_keys)}: {', '.join(result.missing_keys[:10])}")
    if result.unexpected_keys:
        problems.append(f"unexpected {len(result.unexpected_keys)}: {', '.join(result.unexpected_keys[:10])}")
    if problems:
        raise KeyError("Streamed weights do not match the module (" + "; ".join(problems) + ")")


@contextlib.contextmanager
def empty_weights():
    """Build modules with their parameters on the `meta` device.

    Unlike `with torch.device("meta")`, buffers are created normally, so
    non-persistent buffers computed in `__init__` (rotary `inv_freq`, for
    instance), which are never in a checkpoint, keep real values.
    """
    import torch
    from torch import nn

    original = nn.Module.register_parameter

    def register_parameter(self, name, param):
        original(self, name, param)
        if param is not None and param.device.type != "meta":
            kwargs = dict(param.__dict__)
            kwargs["requires_grad"] = param.requires_grad
            self._parameters[name] = type(param)(param.to(torch.device("meta")), **kwargs)

    nn.Module.register_parameter = register_parameter  # type: ignore[method-assign]
    try:
        yield
    finally:
        nn.Module.register_parameter = original  # type: ignore[method-assign]
