"""The @nsys_profile decorator: run a Flyte task under NVIDIA Nsight Systems.

Sits above @env.task. It stamps the task's container so the runtime re-execs the whole action
under `nsys launch`, then wraps execution to bracket the work with `nsys start` / `nsys stop`,
summarize the trace into the task's report, and hand the .nsys-rep back as a trace output.
The user's inputs, body, and return value are untouched.

    from flyteplugins.nsight import nsys_profile

    @nsys_profile(trace=["cuda", "nvtx", "cudnn", "cublas"])
    @env.task
    async def train(epochs: int = 20) -> Checkpoint:
        ...ordinary training code...
        return ckpt
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence, TypeVar, cast

from flyte._task import AsyncFunctionTaskTemplate

from . import _capture, _control, _report

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Session and output are templated with $RUN_NAME/$ACTION_NAME and resolved per action in-container,
# so every action gets a distinct nsys session and report path.
_SESSION_TEMPLATE = "flyte-$RUN_NAME-$ACTION_NAME"
_OUTPUT_TEMPLATE = "/tmp/nsys/$ACTION_NAME/report"


def nsys_profile(
    _task: Optional[F] = None,
    *,
    trace: Sequence[str] = ("cuda", "nvtx"),
    sample: Optional[str] = None,
    capture: str = "task",
    reports: Sequence[str] = _report.DEFAULT_REPORTS,
    attach_report: bool = True,
    enabled: bool = True,
) -> F:
    """Profile a Flyte task with Nsight Systems.

    Args:
        trace: nsys trace domains, e.g. cuda, nvtx, cudnn, cublas, osrt. osrt (OS-runtime) and
            GPU-counter sampling need elevated pod capabilities; see allow_nested_sandboxing().
        sample: CPU sampling mode passed to `nsys -s` (e.g. "cpu", "none"). Omitted by default.
        capture: "task" profiles the whole task body automatically. "manual" launches the task
            under nsys but leaves collection to nsys.range(...) blocks in the body (`async with`
            in an async task, plain `with` in a sync task).
        reports: which `nsys stats` reports to render into the deck.
        attach_report: also surface the .nsys-rep as a downloadable trace output.
        enabled: when False the decorator is a transparent passthrough, so profiling can be kept
            in code and turned off without removing it.

    Decorator order: @nsys_profile must be the outermost decorator, above @env.task.
    """

    def deco(task: F) -> F:
        if not isinstance(task, AsyncFunctionTaskTemplate):
            raise TypeError("@nsys_profile must be applied to a Flyte task (place it above @env.task).")
        if not enabled:
            return task
        if capture not in ("task", "manual"):
            raise ValueError(f"capture must be 'task' or 'manual', got {capture!r}")

        # A ClusteredTaskEnvironment builds a ClusteredTaskTemplate (task_type "clustered-task"); on
        # those, only the primary torchrun worker is profiled. Detected by task_type so the plugin
        # takes no dependency on flyte.clustered.
        clustered = getattr(task, "task_type", None) == "clustered-task"
        container_env = _control.build_container_env(
            session_template=_SESSION_TEMPLATE,
            output_template=_OUTPUT_TEMPLATE,
            trace=trace,
            sample=sample,
            clustered=clustered,
        )
        task = cast(AsyncFunctionTaskTemplate, task.override(env_vars={**(task.env_vars or {}), **container_env}))

        # Metrics need the report deck; enable it so adding the decorator is enough.
        task.report = True

        original_execute = task.execute

        async def wrapped_execute(*args: Any, **kwargs: Any) -> Any:
            # Local run, or the runtime re-exec hook did not fire: run the task unprofiled.
            if not _control.under_nsys():
                import os

                if (
                    os.environ.get("TORCHELASTIC_RUN_ID")
                    and os.environ.get("RANK", "0") == "0"
                    and (_control.nsys_available())
                ):
                    # The primary worker of a clustered task should run under nsys. If it does not,
                    # the runtime re-exec hook never fired — almost always because _F_EXEC_WRAPPER /
                    # _F_EXEC_WRAPPER_CLUSTERED did not reach the worker process.
                    logger.warning(
                        "nsys_profile: primary clustered worker (RANK 0) is not under nsys; running "
                        "unprofiled. The runtime re-exec hook did not fire. Verify _F_EXEC_WRAPPER and "
                        "_F_EXEC_WRAPPER_CLUSTERED are present in this worker's environment."
                    )
                elif not _control.nsys_available():
                    logger.debug("nsys not on PATH; running task unprofiled")
                return await original_execute(*args, **kwargs)

            # Manual mode: the body drives collection via nsys.range(...); just run it.
            if capture == "manual":
                return await original_execute(*args, **kwargs)

            # Whole-task mode: bracket the body with start/stop.
            try:
                report_path = await _control.start_collection()
            except _control.NsysError as e:
                logger.warning("nsys: could not start collection, running unprofiled: %s", e)
                return await original_execute(*args, **kwargs)

            try:
                result = await original_execute(*args, **kwargs)
            finally:
                # Finalize even if the body raised — profiling a crash is often the point.
                try:
                    await _control.stop_collection()
                    summary = await _capture.finalize(report_path, reports=reports, attach=attach_report)
                    if summary:
                        logger.info("nsys profile summary: %s", summary)
                except _control.NsysError as e:
                    logger.warning("nsys: could not finalize collection: %s", e)

            return result

        task.execute = wrapped_execute
        return cast(F, task)

    if _task is not None:
        return deco(_task)
    return deco
