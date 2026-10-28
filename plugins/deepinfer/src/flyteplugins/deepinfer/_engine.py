"""`DeepInfer`: a lazily started, persistent pool of DeepInfer engines.

DeepInfer runs one engine *process* per data-parallel rank, each pinned to its own
devices, with the MoE experts shared across all of them. `DeepInfer` owns that pool:

- Constructing it is cheap and touches no GPU, so it can live at module level.
- The first `generate` call spawns this node's engine processes and waits until they
  have loaded the model. They then stay up for the life of the task process, so on a
  reusable container every later call (and every later task invocation) skips the
  model load.
- `generate` strides the prompts across the engines, applies the sampling and
  structured-output options, decodes, and returns one result per prompt in input order.

On a multi-node (`ClusteredTaskEnvironment`) job every node runs the same task body:
the prompts are split across the engines of *all* nodes, and each node's `generate`
fills in only the results its own engines produced (the rest are `None`).
"""

from __future__ import annotations

import asyncio
import atexit
import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from flyteplugins.deepinfer import _constants as const
from flyteplugins.deepinfer._sampling import build_sampling

if TYPE_CHECKING:
    import flyte.io

# How often the parent re-checks engine liveness while it waits on results.
_POLL_SECONDS = 1.0


@dataclass(frozen=True)
class _EngineConfig:
    """Everything an engine process needs; must stay picklable."""

    model: str
    ep_size: int
    cp_size: int
    env: dict[str, str] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


class DeepInfer:
    """Offline DeepInfer inference over every GPU visible to the task.

    ```python
    llm = DeepInfer("deepseek-ai/DeepSeek-V4.1-Flash-1017")

    @env.task
    async def score(prompts: list[str]) -> list[str]:
        return await llm.generate.aio(prompts, temperature=0, max_tokens=512)
    ```

    Args:
        model: Hugging Face model id, local checkpoint directory, or a `flyte.io.Dir`
            holding a checkpoint (downloaded on first use).
        ep_size: Expert-parallel size. Defaults to every device in the job
            (`WORLD_SIZE` nodes x devices on this node).
        cp_size: Devices per engine (context parallelism). The default of 1 runs one
            engine per device, which maximizes throughput.
        env: Extra environment variables for the engine processes, applied before
            `import deepinfer`. They take precedence over the plugin's runtime defaults
            (e.g. `{"DEEPINFER_USE_NEXTN_DECODE": "0"}` turns off speculative decoding).
        **engine_kwargs: Forwarded to DeepInfer's `LLM` (for example
            `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`).
    """

    def __init__(
        self,
        model: Union[str, Path, "flyte.io.Dir"],
        *,
        ep_size: Optional[int] = None,
        cp_size: int = 1,
        env: Optional[dict[str, str]] = None,
        **engine_kwargs: Any,
    ) -> None:
        if cp_size < 1:
            raise ValueError("cp_size must be >= 1")
        self._model = model
        self._ep_size = ep_size
        self._cp_size = cp_size
        self._env = dict(env or {})
        self._engine_kwargs = engine_kwargs
        self._lock = threading.Lock()
        self._pool: Optional[_EnginePool] = None

    @property
    def generate(self) -> _Generate:
        """Generate one completion per prompt, in input order.

        Call it as `llm.generate(prompts, ...)` from synchronous code, or
        `await llm.generate.aio(prompts, ...)` inside an async task. See `_Generate`
        for the options.
        """
        return _Generate(self)

    def close(self) -> None:
        """Stop the engine processes and free their GPUs. Safe to call repeatedly."""
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def __enter__(self) -> DeepInfer:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _run(self, prompts: list[str], model: Optional[str], options: dict[str, Any]) -> list[Any]:
        """Blocking core of `generate`; `model` is the resolved path, needed only before the first start."""
        params, parse = build_sampling(**options)
        if not prompts:
            return []
        # Engines handle one request at a time; the lock also serializes the first start,
        # so concurrent first calls never launch two pools.
        with self._lock:
            if self._pool is None:
                config = _EngineConfig(
                    model=model if model is not None else _resolve_model_sync(self._model),
                    ep_size=self._ep_size or _default_ep_size(),
                    cp_size=self._cp_size,
                    env=self._env,
                    engine_kwargs=self._engine_kwargs,
                )
                self._pool = _EnginePool.start(config)
                # Runs before multiprocessing's own exit hook joins the engine processes.
                atexit.register(self.close)
            try:
                texts = self._pool.generate(list(prompts), params)
            except BaseException:
                # An engine died or errored: the pool is no longer usable, so tear it down
                # rather than leave a broken one cached for the next call to hang on.
                self._pool.close()
                self._pool = None
                raise
        return [None if text is None else parse(text) for text in texts]


class _Generate:
    """`DeepInfer.generate`: call it directly from sync code, or `await` its `.aio` from async code.

    Unset sampling options fall back to DeepInfer's own `SamplingParams` defaults; note
    that upstream's default `max_tokens` is small. At most one of `json_schema`, `regex`
    and `grammar` may be set to constrain decoding.

    Args:
        prompts: Prompts to complete.
        temperature: Sampling temperature (`0` for greedy).
        top_p: Nucleus sampling probability mass.
        top_k: Top-k cutoff.
        max_tokens: Maximum tokens to generate per prompt.
        json_schema: Constrain output to a JSON schema: a Pydantic model class (results
            are validated instances), or a dict / JSON string schema (results are parsed
            with `json.loads`).
        regex: Constrain output to match this regular expression.
        grammar: Constrain output to this EBNF grammar.
        **sampling_kwargs: Any other DeepInfer `SamplingParams` field.

    Returns:
        One result per prompt. On a multi-node job, entries for prompts handled by other
        nodes' engines are `None`.
    """

    def __init__(self, llm: DeepInfer) -> None:
        self._llm = llm

    def __call__(
        self,
        prompts: list[str],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        json_schema: Any = None,
        regex: Optional[str] = None,
        grammar: Optional[str] = None,
        **sampling_kwargs: Any,
    ) -> list[Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("generate() blocks; inside an async function use `await llm.generate.aio(...)`")
        options = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            json_schema=json_schema,
            regex=regex,
            grammar=grammar,
            **sampling_kwargs,
        )
        return self._llm._run(prompts, None, options)

    async def aio(
        self,
        prompts: list[str],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        json_schema: Any = None,
        regex: Optional[str] = None,
        grammar: Optional[str] = None,
        **sampling_kwargs: Any,
    ) -> list[Any]:
        options = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            json_schema=json_schema,
            regex=regex,
            grammar=grammar,
            **sampling_kwargs,
        )
        # Resolve a `flyte.io.Dir` here, in the task's own context; then wait for the
        # engines off the event loop (`to_thread` carries the context along).
        model = await _resolve_model(self._llm._model) if self._llm._pool is None and prompts else None
        return await asyncio.to_thread(self._llm._run, prompts, model, options)


async def _resolve_model(model: Any) -> str:
    """Turn a `flyte.io.Dir` into a local path; leave ids and paths to the engine."""
    download = getattr(model, "download", None)
    if download is not None:
        return await download()
    return os.fspath(model)


def _resolve_model_sync(model: Any) -> str:
    download_sync = getattr(model, "download_sync", None)
    if download_sync is not None:
        return download_sync()
    return os.fspath(model)


def _default_ep_size() -> int:
    from flyteplugins.deepinfer._bootstrapping import get_actual_num_gpus

    return int(os.environ.get(const.WORLD_SIZE, "1")) * get_actual_num_gpus()


class _EnginePool:
    """This node's engine processes and the queues that drive them."""

    def __init__(self, dp_size: int, local_ranks: list[int], procs: list, requests: dict, responses: Any) -> None:
        self.dp_size = dp_size
        self.local_ranks = local_ranks
        self._procs = procs
        self._requests = requests
        self._responses = responses
        self._next_id = 0

    @classmethod
    def start(cls, config: _EngineConfig) -> _EnginePool:
        dp_size = max(config.ep_size // config.cp_size, 1)
        num_nodes = int(os.environ.get(const.WORLD_SIZE, "1"))
        node_rank = int(os.environ.get(const.RANK, "0"))
        if dp_size % num_nodes:
            raise ValueError(
                f"{dp_size} engines (ep_size={config.ep_size} / cp_size={config.cp_size}) do not split evenly "
                f"across {num_nodes} nodes"
            )
        per_node = dp_size // num_nodes
        local_ranks = list(range(node_rank * per_node, (node_rank + 1) * per_node))

        # `spawn`: each engine must import torch/deepinfer fresh, after its devices are pinned.
        ctx = get_context("spawn")
        responses = ctx.Queue()
        requests = {rank: ctx.Queue() for rank in local_ranks}
        procs = [
            ctx.Process(
                target=_engine_main,
                args=(config, dp_size, rank, requests[rank], responses),
                name=f"deepinfer-engine-{rank}",
                # Not daemonic: daemonic processes cannot start children of their own.
                daemon=False,
            )
            for rank in local_ranks
        ]
        for p in procs:
            p.start()
        pool = cls(dp_size, local_ranks, procs, requests, responses)
        try:
            pool._collect(expected=set(local_ranks), kind="ready")
        except BaseException:
            pool.close()
            raise
        return pool

    def generate(self, prompts: list[str], params: dict[str, Any]) -> list[Optional[str]]:
        self._next_id += 1
        req_id = self._next_id
        # Global prompt i goes to engine i % dp_size, so every node agrees on the split.
        assignments = {rank: list(range(rank, len(prompts), self.dp_size)) for rank in self.local_ranks}
        for rank in self.local_ranks:
            indices = assignments[rank]
            # Expert parallelism couples the engines, so each one must take part in every
            # call: an engine with no prompts of its own runs a throwaway copy of one.
            shard = [prompts[i] for i in indices] if indices else prompts[:1]
            self._requests[rank].put((req_id, shard, params))

        results: list[Optional[str]] = [None] * len(prompts)
        for rank, texts in self._collect(expected=set(self.local_ranks), kind="result", req_id=req_id).items():
            for i, text in zip(assignments[rank], texts):
                results[i] = text
        return results

    def close(self) -> None:
        for q in self._requests.values():
            try:
                q.put(None)
            except Exception:  # queue already torn down
                pass
        for p in self._procs:
            p.join(timeout=30)
        for p in self._procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    def _collect(self, *, expected: set[int], kind: str, req_id: Optional[int] = None) -> dict[int, Any]:
        """Wait for one `kind` message per expected rank, failing fast if an engine dies."""
        got: dict[int, Any] = {}
        while len(got) < len(expected):
            try:
                msg_kind, rank, msg_req, payload = self._responses.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                dead = [r for r, p in zip(self.local_ranks, self._procs) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"DeepInfer engine(s) {dead} exited unexpectedly")
                continue
            if msg_kind == "error":
                raise RuntimeError(f"DeepInfer engine {rank} failed:\n{payload}")
            if msg_kind == kind and msg_req == req_id:
                got[rank] = payload
        return got


def _engine_main(config: _EngineConfig, dp_size: int, dp_rank: int, requests: Any, responses: Any) -> None:
    """Entry point of one engine process: build the engine, then serve requests until told to stop."""
    try:
        from flyteplugins.deepinfer._bootstrapping import create_llm, setup_common_env

        os.environ.update(config.env)  # user values first: setup_common_env only fills gaps
        setup_common_env()
        llm = create_llm(config.model, dp_rank, ep_size=config.ep_size, cp_size=config.cp_size, **config.engine_kwargs)
        from deepinfer.sampling_params import SamplingParams

        tokenizer = llm.get_tokenizer()
    except BaseException:
        responses.put(("error", dp_rank, None, traceback.format_exc()))
        return
    responses.put(("ready", dp_rank, None, None))

    while True:
        request = requests.get()
        if request is None:
            break
        req_id, prompts, params = request
        try:
            outputs = llm.generate(
                prompts=prompts,
                sampling_params=[SamplingParams(**params) for _ in prompts],
                use_tqdm=False,
            )
            # Each output is a list of per-token results; special tokens (EOS) are dropped.
            texts = [tokenizer.decode([tok.token_id for tok in out], skip_special_tokens=True) for out in outputs]
            responses.put(("result", dp_rank, req_id, texts))
        except BaseException:
            responses.put(("error", dp_rank, req_id, traceback.format_exc()))
    close = getattr(llm, "close", None)
    if close is not None:
        close()
