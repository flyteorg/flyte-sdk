"""Worker entrypoint for the multiprocessing local controller.

This module is imported in fresh subprocess interpreters by
``ProcessPoolExecutor`` (spawn). It must be import-safe (no side effects)
and ``_worker_run`` must remain at module level so it is picklable.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle  # type: ignore[import-untyped]

from flyte._context import Context, internal_ctx
from flyte._internal.runtime import convert
from flyte._internal.runtime.entrypoints import direct_dispatch
from flyte._logging import logger
from flyte._task import TaskTemplate
from flyte.models import ActionID, CheckpointPaths, CodeBundle, RawDataPath, TaskContext

_PARENT_WATCHDOG_INTERVAL_SEC = 1.0


@dataclass
class _WorkerPayload:
    """Cloudpickled payload sent from parent to worker."""

    task: TaskTemplate
    action: ActionID
    raw_data_path: RawDataPath
    inputs: convert.Inputs
    version: str
    checkpoint_paths: CheckpointPaths | None
    code_bundle: CodeBundle | None
    output_path: str
    run_base_dir: str
    parent_mode: str  # "local" — workers always run as local sub-controllers
    disable_run_cache: bool
    custom_context: dict[str, str]
    init_snapshot: dict[str, Any] | None  # picklable subset of parent's _InitConfig


def _parent_pid_watchdog(parent_pid: int) -> None:
    """Daemon thread: if the parent process dies (so getppid() returns 1
    on POSIX or some other value), exit this worker immediately so we
    don't become a long-lived orphan after a parent SIGKILL."""
    while True:
        time.sleep(_PARENT_WATCHDOG_INTERVAL_SEC)
        try:
            current_ppid = os.getppid()
        except Exception:
            os._exit(0)
        if current_ppid != parent_pid:
            # Parent gone — was reparented to init (1 on Linux, launchd on macOS).
            os._exit(0)


def _worker_initializer(parent_pid: int) -> None:
    """Run once per worker process when ProcessPoolExecutor spawns it.

    Workers ignore SIGINT so terminal Ctrl-C is owned by the parent —
    the parent's signal handler decides when to terminate the pool.
    Without this, every worker would dump a KeyboardInterrupt traceback
    on every Ctrl-C, racing with the parent's shutdown.

    A daemon thread watches the parent's pid; if the parent vanishes
    (e.g. SIGKILL), the worker self-exits to avoid orphans.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass

    if parent_pid > 0 and sys.platform != "win32":
        t = threading.Thread(
            target=_parent_pid_watchdog,
            args=(parent_pid,),
            name="flyte-parent-watchdog",
            daemon=True,
        )
        t.start()


def _ensure_init(snapshot: dict[str, Any] | None) -> None:
    """Reconstruct a minimal ``_InitConfig`` in this worker process so that
    code paths gated by ``requires_initialization`` find a config. We strip
    the gRPC client and Storage objects from the parent's snapshot — the
    worker only does local execution and does not need them."""
    from flyte import _initialize

    if _initialize._init_config is not None:
        return

    root_dir = Path(snapshot["root_dir"]) if snapshot and snapshot.get("root_dir") else Path.cwd()
    _initialize._init_config = _initialize._InitConfig(
        root_dir=root_dir,
        org=snapshot.get("org") if snapshot else None,
        project=snapshot.get("project") if snapshot else None,
        domain=snapshot.get("domain") if snapshot else None,
        batch_size=snapshot.get("batch_size", 1000) if snapshot else 1000,
        sync_local_sys_paths=snapshot.get("sync_local_sys_paths", True) if snapshot else True,
        local_persistence=False,  # never persist from workers; parent owns the DB
        client=None,
        storage=None,
        image_builder=snapshot.get("image_builder", "local") if snapshot else "local",
        images=snapshot.get("images", {}) if snapshot else {},
    )


def _ensure_local_controller():
    """Install a fresh in-process LocalController so any nested task calls
    inside the worker dispatch in-process (no recursive multiprocessing)."""
    from flyte._internal.controllers import _ControllerState, create_controller

    if _ControllerState.controller is None:
        create_controller("local")


def _build_task_context(p: _WorkerPayload) -> TaskContext:
    import flyte.report

    return TaskContext(
        action=p.action,
        version=p.version,
        raw_data_path=p.raw_data_path,
        output_path=p.output_path,
        run_base_dir=p.run_base_dir,
        report=flyte.report.Report(name=p.action.name),
        checkpoint_paths=p.checkpoint_paths,
        code_bundle=p.code_bundle,
        mode=p.parent_mode,  # type: ignore[arg-type]
        custom_context=p.custom_context,
        disable_run_cache=p.disable_run_cache,
    )


def _worker_run(payload_bytes: bytes) -> bytes:
    """Top-level worker entrypoint. Cloudpickle in, cloudpickle out.

    Returns ``cloudpickle.dumps((Outputs|None, Error|None))``.
    """
    try:
        p: _WorkerPayload = cloudpickle.loads(payload_bytes)

        _ensure_init(p.init_snapshot)
        _ensure_local_controller()

        from flyte._internal.controllers import get_controller
        from flyte._run import _run_mode_var

        _run_mode_var.set("local")

        controller = get_controller()
        tctx = _build_task_context(p)
        ctx = internal_ctx()
        new_ctx = Context(ctx.data.replace(task_context=tctx, tracker=None))

        async def _go() -> tuple[Any, Any]:
            with new_ctx:
                return await direct_dispatch(
                    p.task,
                    controller=controller,
                    action=p.action,
                    raw_data_path=p.raw_data_path,
                    inputs=p.inputs,
                    version=p.version,
                    checkpoint_paths=p.checkpoint_paths,
                    code_bundle=p.code_bundle,
                    output_path=p.output_path,
                    run_base_dir=p.run_base_dir,
                )

        out, err = asyncio.run(_go())
        return cloudpickle.dumps((out, err))
    except BaseException as exc:
        # Convert to an Error so the parent can render it identically to
        # an in-process failure. Re-raising would lose traceback fidelity
        # across the process boundary because ProcessPoolExecutor pickles
        # the exception class but not arbitrary state.
        logger.exception("worker_run raised in pid=%s: %s", os.getpid(), exc)
        try:
            err = convert.convert_from_native_to_error(exc)
        except Exception:
            from flyteidl2.core import execution_pb2

            err = convert.Error(
                err=execution_pb2.ExecutionError(
                    code="WorkerFailure",
                    message=f"{type(exc).__name__}: {exc}",
                ),
                recoverable=False,
            )
        return cloudpickle.dumps((None, err))
