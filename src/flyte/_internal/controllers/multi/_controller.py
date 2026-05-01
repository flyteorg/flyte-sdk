"""Multiprocessing variant of :class:`LocalController`.

Selected via mode ``"local-multi"`` (CLI flag ``--local-multi``). Each call
to :meth:`submit` orchestrates the same way the in-process LocalController
does (cache lookup, recorder lifecycle, retries) but executes the task
body in a worker process from a :class:`ProcessPoolExecutor`. This gives
real CPU parallelism for fan-out workflows on a single machine.

Workers do not touch the Textual tracker, the run-history SQLite DB, or
the ``LocalTaskCache`` — those stay parent-only. Only the
``(task, inputs, paths)`` payload and the ``(Outputs, Error)`` return
value cross the process boundary, both via cloudpickle.
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing
import os
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import cloudpickle  # type: ignore[import-untyped]
from flyteidl2.core import execution_pb2

from flyte._internal.runtime import convert
from flyte._internal.runtime.convert import Error, Inputs, Outputs
from flyte._logging import logger
from flyte._persistence._task_cache import LocalTaskCache
from flyte._task import TaskTemplate
from flyte.models import ActionID, CheckpointPaths, CodeBundle, RawDataPath

from .._local_controller import LocalController
from ._worker import _worker_initializer, _worker_run, _WorkerPayload

_ENV_MAX_WORKERS = "FLYTE_LOCAL_MULTI_WORKERS"
_DEFAULT_MAX_WORKERS = min(os.cpu_count() or 1, 8)
_TERMINATE_GRACE_SEC = 1.5

# Module-level registry of live controllers so a single signal handler
# (installed once on the main thread) can shut all of them down on
# SIGINT/SIGTERM. This indirection is necessary because controllers are
# constructed on the syncify background thread, where ``signal.signal``
# cannot be called.
_active_controllers: "list[LocalMultiController]" = []
_active_lock = threading.Lock()
_signals_installed = False
_prev_sigint = None
_prev_sigterm = None


def _module_signal_handler(signum, frame):
    """Synchronously terminate all live controllers' worker pools, then
    defer to the user's previous handler so the process exits as
    expected. Runs on the main thread."""
    logger.warning("local-multi: received signal %d, terminating worker pools", signum)
    with _active_lock:
        controllers = list(_active_controllers)
    for ctrl in controllers:
        try:
            ctrl._shutdown_pool(force=True)
        except Exception as exc:
            logger.debug("signal-driven shutdown raised: %s", exc)

    prev = _prev_sigint if signum == signal.SIGINT else _prev_sigterm
    if prev in (None, signal.SIG_DFL):
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        os._exit(128 + signum)
    if prev == signal.SIG_IGN:
        return
    if callable(prev):
        prev(signum, frame)


def install_signal_handlers() -> None:
    """Idempotently install SIGINT/SIGTERM handlers from the main thread.

    Must be called from the main thread (``signal.signal`` is restricted
    to it). Safe to call multiple times. The natural call site is from
    ``_Runner.__init__`` when mode is ``"local-multi"``, since that
    constructor runs on the user's main thread before the syncify
    boundary."""
    global _signals_installed, _prev_sigint, _prev_sigterm  # noqa: PLW0603
    if _signals_installed:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        _prev_sigint = signal.signal(signal.SIGINT, _module_signal_handler)
        _prev_sigterm = signal.signal(signal.SIGTERM, _module_signal_handler)
        _signals_installed = True
    except (ValueError, OSError) as exc:
        logger.debug("install_signal_handlers failed: %s", exc)


def prewarm_resource_tracker() -> None:
    """Force-spawn the multiprocessing resource_tracker subprocess now, while
    the real ``sys.stderr`` is still installed.

    Lazy startup happens later (inside ``ProcessPoolExecutor`` worker spawn)
    via ``resource_tracker.ensure_running()``, which builds its child's
    ``pass_fds`` from ``sys.stderr.fileno()``. A TUI like Textual replaces
    ``sys.stderr`` with a capture stream whose ``fileno()`` returns ``-1``,
    and ``_posixsubprocess.fork_exec`` rejects negative fds with
    ``ValueError('bad value(s) in fds_to_keep')``. Pre-warming sidesteps
    this: once the tracker is up, subsequent spawns reuse its existing fd
    and never touch ``sys.stderr`` again. Must be called from the parent
    process before TUI initialization."""
    try:
        from multiprocessing import resource_tracker

        resource_tracker._resource_tracker.ensure_running()
    except Exception as exc:
        logger.debug("prewarm_resource_tracker failed: %s", exc)


def _register_controller(ctrl: "LocalMultiController") -> None:
    with _active_lock:
        _active_controllers.append(ctrl)


def _unregister_controller(ctrl: "LocalMultiController") -> None:
    with _active_lock:
        try:
            _active_controllers.remove(ctrl)
        except ValueError:
            pass


def _build_init_snapshot() -> dict[str, Any] | None:
    """Capture the picklable subset of the parent's ``_InitConfig`` so the
    worker can re-create a compatible config without needing the gRPC
    client or storage backend (which are typically not picklable)."""
    from flyte import _initialize

    cfg = _initialize._init_config
    if cfg is None:
        return None
    return {
        "root_dir": str(cfg.root_dir),
        "org": cfg.org,
        "project": cfg.project,
        "domain": cfg.domain,
        "batch_size": cfg.batch_size,
        "sync_local_sys_paths": cfg.sync_local_sys_paths,
        "image_builder": cfg.image_builder,
        "images": dict(cfg.images),
    }


class LocalMultiController(LocalController):
    """LocalController that dispatches each task body to a subprocess pool."""

    def __init__(self, max_workers: int | None = None):
        super().__init__()
        env_workers = os.environ.get(_ENV_MAX_WORKERS)
        if max_workers is None and env_workers:
            try:
                max_workers = int(env_workers)
            except ValueError:
                logger.warning("Invalid %s=%r; falling back to default", _ENV_MAX_WORKERS, env_workers)
        self._max_workers = max_workers or _DEFAULT_MAX_WORKERS
        self._pool: ProcessPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._mp_ctx = multiprocessing.get_context("spawn")
        self._stopped = False
        # The action name of the top-level run. The root task's body runs
        # in-process so that its `await child(...)` sub-task calls can fan
        # out to multiple worker processes; only the sub-tasks themselves
        # are dispatched to the pool.
        self._root_action_name: str | None = None
        # Best-effort cleanup if the user forgets to await stop() (e.g. on
        # an unhandled exception path). We register a weakref-friendly
        # cleanup that terminates worker processes synchronously.
        atexit.register(self._atexit_cleanup)
        _register_controller(self)
        logger.debug("LocalMultiController init max_workers=%d", self._max_workers)

    def set_root_action(self, action_name: str) -> None:
        """Called by ``_run_local`` so the controller knows which submit is
        the root invocation (run in-process) vs. nested sub-task dispatch
        (run in a worker)."""
        self._root_action_name = action_name

    def _get_pool(self) -> ProcessPoolExecutor:
        with self._pool_lock:
            if self._stopped:
                raise RuntimeError("LocalMultiController has been stopped; no new submits allowed.")
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self._max_workers,
                    mp_context=self._mp_ctx,
                    initializer=_worker_initializer,
                    initargs=(os.getpid(),),
                )
            return self._pool

    async def _run_dispatch(
        self,
        _task: TaskTemplate,
        *,
        action: ActionID,
        raw_data_path: RawDataPath,
        inputs: Inputs,
        version: str,
        checkpoint_paths: CheckpointPaths | None,
        code_bundle: CodeBundle | None,
        output_path: str,
        run_base_dir: str,
    ) -> tuple[Outputs | None, Error | None]:
        from flyte._context import internal_ctx
        from flyte._internal.runtime.entrypoints import direct_dispatch

        ctx = internal_ctx()
        parent_tctx = ctx.data.task_context
        disable_run_cache = parent_tctx.disable_run_cache if parent_tctx else False
        custom_context = dict(parent_tctx.custom_context) if parent_tctx and parent_tctx.custom_context else {}

        # Root task body runs in the parent process so its nested
        # `controller.submit(...)` calls can fan out across the pool.
        # Sub-tasks (parent action ≠ root) are dispatched to workers.
        is_root = parent_tctx is not None and parent_tctx.action.name == self._root_action_name
        if is_root:
            return await direct_dispatch(
                _task,
                controller=self,
                action=action,
                raw_data_path=raw_data_path,
                inputs=inputs,
                version=version,
                checkpoint_paths=checkpoint_paths,
                code_bundle=code_bundle,
                output_path=output_path,
                run_base_dir=run_base_dir,
            )

        payload = _WorkerPayload(
            task=_task,
            action=action,
            raw_data_path=raw_data_path,
            inputs=inputs,
            version=version,
            checkpoint_paths=checkpoint_paths,
            code_bundle=code_bundle,
            output_path=output_path,
            run_base_dir=run_base_dir,
            parent_mode="local",  # workers always behave as local sub-controllers
            disable_run_cache=disable_run_cache,
            custom_context=custom_context,
            init_snapshot=_build_init_snapshot(),
        )

        try:
            payload_bytes = cloudpickle.dumps(payload)
        except Exception as exc:
            logger.error("Failed to cloudpickle task payload for '%s': %s", _task.name, exc)
            return None, convert.convert_from_native_to_error(exc)

        loop = asyncio.get_running_loop()
        try:
            cf_future = self._get_pool().submit(_worker_run, payload_bytes)
        except BrokenProcessPool:
            self._reset_pool_on_broken()
            return None, _broken_pool_error("submission rejected")
        except RuntimeError as exc:
            # Pool was stopped concurrently with this submit.
            return None, _broken_pool_error(str(exc))

        try:
            result_bytes = await asyncio.wrap_future(cf_future, loop=loop)
        except asyncio.CancelledError:
            cf_future.cancel()
            raise
        except BrokenProcessPool as exc:
            # A worker died (OOM, segfault, os._exit, etc.) while running
            # this task. The pool is now poisoned; reset it so subsequent
            # retries / submissions get a fresh pool.
            logger.warning("Worker died during task '%s': %s", _task.name, exc)
            self._reset_pool_on_broken()
            return None, _broken_pool_error(f"worker died: {exc}")

        out, err = cloudpickle.loads(result_bytes)
        return out, err

    def _reset_pool_on_broken(self) -> None:
        """Discard a poisoned ProcessPoolExecutor so the next submit gets a
        fresh pool. The broken pool's threads/queues clean themselves up
        when garbage collected; we don't wait."""
        with self._pool_lock:
            if self._pool is not None:
                broken = self._pool
                self._pool = None
                try:
                    broken.shutdown(wait=False, cancel_futures=True)
                except Exception as exc:
                    logger.debug("broken pool shutdown raised: %s", exc)

    async def stop(self):
        await LocalTaskCache.close()
        self._shutdown_pool(force=True)
        _unregister_controller(self)

    def _shutdown_pool(self, *, force: bool) -> None:
        """Tear the pool down. SIGTERM each worker, brief join, then
        SIGKILL any survivor. Leaf-task ``try/finally`` blocks are NOT
        guaranteed to run on Ctrl-C — by design, the priority is exiting
        promptly with no orphans, not graceful in-task cleanup."""
        with self._pool_lock:
            self._stopped = True
            pool, self._pool = self._pool, None

        if pool is None:
            return

        # Snapshot worker processes BEFORE shutdown so we can terminate
        # them. ProcessPoolExecutor exposes them via the private
        # ``_processes`` dict; guard with getattr in case the internal
        # API changes between Python versions.
        workers = list(getattr(pool, "_processes", {}).values())

        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            logger.debug("pool.shutdown raised (continuing to terminate workers): %s", exc)

        for proc in workers:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception as exc:
                logger.debug("terminate(pid=%s) raised: %s", getattr(proc, "pid", "?"), exc)

        for proc in workers:
            try:
                proc.join(timeout=_TERMINATE_GRACE_SEC)
            except Exception as exc:
                logger.debug("join(pid=%s) raised: %s", getattr(proc, "pid", "?"), exc)

        for proc in workers:
            try:
                if proc.is_alive():
                    logger.warning("Worker pid=%s did not exit on SIGTERM, SIGKILLing", proc.pid)
                    proc.kill()
                    proc.join(timeout=_TERMINATE_GRACE_SEC)
            except Exception as exc:
                logger.debug("kill(pid=%s) raised: %s", getattr(proc, "pid", "?"), exc)

    def _atexit_cleanup(self) -> None:
        """Last-ditch cleanup if the run path skipped ``stop()`` (e.g. a
        crash before ``_run_local``'s finally block ran). Always sync."""
        if self._pool is not None:
            try:
                self._shutdown_pool(force=False)
            except Exception as exc:
                logger.debug("atexit cleanup raised: %s", exc)


def _broken_pool_error(detail: str) -> Error:
    """Recoverable error so the controller's retry loop gets another chance
    on a fresh pool. The user sees a normal task error, not a stack trace
    from concurrent.futures internals."""
    return Error(
        err=execution_pb2.ExecutionError(
            code="WorkerProcessDied",
            message=f"local-multi worker process died unexpectedly ({detail}).",
        ),
        recoverable=True,
    )
