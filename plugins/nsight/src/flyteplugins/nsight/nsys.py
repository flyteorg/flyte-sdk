"""Region profiling: collect only part of a task.

Whole-task profiling of a long run produces an unwieldy multi-gigabyte trace. When you only
want the hot loop, put the task under nsys with `@nsys_profile(capture="manual")` and wrap the
region you care about. The region matches your task body — `async with` in an `async def` task,
plain `with` in a `def` task, since `@nsys_profile` accepts both:

    from flyteplugins.nsight import nsys, nvtx

    @nsys_profile(capture="manual")
    @env.task(report=True)
    async def train():                       # async task -> `async with`
        warmup()
        async with nsys.range("hot-loop"):
            for step in range(100):
                with nvtx.range("step"):
                    train_step()

    @nsys_profile(capture="manual")
    @env.task(report=True)
    def train_sync():                        # sync task -> plain `with`
        warmup()
        with nsys.range("hot-loop"):
            for step in range(100):
                with nvtx.range("step"):
                    train_step()

Each region collects independently, writes its own .nsys-rep, renders its own report section, and
attaches its own trace output. Outside a profiling run (local execution, or the task was not
launched under nsys) the region is a transparent no-op, so the same code runs anywhere.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from . import _capture, _control, _report

logger = logging.getLogger(__name__)


class _Region:
    """A profiling region usable with either `with` (sync task body) or `async with` (async task body).

    `@nsys_profile` accepts both async and sync task functions, so the region it hands back has to
    work in both. The two protocols do the same three things — `nsys start`, run the block, then
    `nsys stop` + summarize — and differ only in whether those calls are awaited; the sync path uses
    the sync-native flyte primitives (blocking subprocess, File.from_local_sync, the sync report
    flush). Outside a profiling run the region is a transparent no-op.
    """

    def __init__(self, name: str, *, reports: Sequence[str], attach: bool) -> None:
        self._name = name
        self._reports = reports
        self._attach = attach
        # Set only once collection actually started. Left None when the block runs unprofiled (not
        # under nsys, or start failed), which tells __exit__/__aexit__ there is nothing to finalize.
        self._report_path: Optional[str] = None

    def _under_nsys(self) -> bool:
        if not _control.under_nsys():
            logger.debug(
                "nsys.range(%s): not running under nsys; skipping collection",
                self._name,
            )
            return False
        return True

    # sync protocol: `with nsys.range(...)` in a non-async task body
    def __enter__(self) -> "_Region":
        if not self._under_nsys():
            return self
        try:
            self._report_path = _control.start_collection_sync(self._name)
        except _control.NsysError as e:
            logger.warning("nsys.range(%s): could not start collection: %s", self._name, e)
        return self

    def __exit__(self, *exc: object) -> bool:
        if self._report_path is None:  # never started; nothing to finalize
            return False
        try:
            _control.stop_collection_sync()
            _capture.finalize_sync(
                self._report_path,
                title=f"Nsight region: {self._name}",
                reports=self._reports,
                attach=self._attach,
            )
        except _control.NsysError as e:
            logger.warning("nsys.range(%s): could not finalize collection: %s", self._name, e)
        return False  # never suppress the body's exception; profiling a crash is often the point

    # async protocol: `async with nsys.range(...)` in an async task body
    async def __aenter__(self) -> "_Region":
        if not self._under_nsys():
            return self
        try:
            self._report_path = await _control.start_collection(self._name)
        except _control.NsysError as e:
            logger.warning("nsys.range(%s): could not start collection: %s", self._name, e)
        return self

    async def __aexit__(self, *exc: object) -> bool:
        if self._report_path is None:  # never started; nothing to finalize
            return False
        try:
            await _control.stop_collection()
            await _capture.finalize(
                self._report_path,
                title=f"Nsight region: {self._name}",
                reports=self._reports,
                attach=self._attach,
            )
        except _control.NsysError as e:
            logger.warning("nsys.range(%s): could not finalize collection: %s", self._name, e)
        return False  # never suppress the body's exception; profiling a crash is often the point


def range(name: str, *, reports: Sequence[str] = _report.DEFAULT_REPORTS, attach: bool = True) -> _Region:
    """Profile the wrapped block as a named region. No-op if not running under nsys.

    Works both ways, matching your task body:

        async with nsys.range("hot-loop"):   # in an `async def` task
            ...
        with nsys.range("hot-loop"):          # in a plain `def` task
            ...
    """
    return _Region(name, reports=reports, attach=attach)


# Alias: reads well as `with nsys.profile("hot-loop"):` / `async with nsys.profile("hot-loop"):`
profile = range
