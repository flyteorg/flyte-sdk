"""Shared finalize path: summarize a finished collection and hand back the trace.

Both the whole-task decorator and the region context manager end the same way — a .nsys-rep on
disk that needs summarizing into the report and surfacing as a downloadable output. This keeps
that in one place. The .nsys-rep is returned through a @flyte.trace function so it shows up as a
trace output in the Flyte UI without changing the task's own return type.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import flyte
from flyte.io import File

from . import _report

logger = logging.getLogger(__name__)


def _exists(path: str) -> bool:
    # Kept sync so the async finalize does not call os.path directly (a one-shot local stat).
    return bool(path) and os.path.exists(path)


@flyte.trace
async def capture_report_file(report_path: str) -> File:
    """Upload the .nsys-rep and surface it as a trace output. Open it in the Nsight Systems GUI."""
    return await File.from_local(report_path)


@flyte.trace
def capture_report_file_sync(report_path: str) -> File:
    """Blocking twin of capture_report_file, for a `with nsys.range(...)` block in a non-async task body."""
    return File.from_local_sync(report_path)


async def finalize(
    report_path: str,
    *,
    title: Optional[str] = None,
    reports: Sequence[str] = _report.DEFAULT_REPORTS,
    attach: bool = True,
) -> dict:
    """Render metrics for a finished collection and (optionally) attach the trace file.

    Best-effort throughout: a task must not fail because its profile could not be summarized.
    Returns a small summary dict for logging.
    """
    if not _exists(report_path):
        logger.warning("nsys report not found at %s; skipping summary", report_path)
        return {}

    # Surface the .nsys-rep. A normal task attaches it as a downloadable File trace output (recorded
    # via the controller). A clustered/jobset worker has no controller, so upload it to durable
    # storage directly and link it from the report deck instead — the upload runs before render so
    # the link lands in the deck.
    clustered = bool(os.environ.get("TORCHELASTIC_RUN_ID"))
    trace_url: Optional[str] = None
    if attach and clustered:
        try:
            trace_url = await _persist_trace(report_path)
        except Exception:
            logger.exception("nsys: failed to persist .nsys-rep for clustered worker")

    summary: dict = {}
    try:
        summary = await _report.render(report_path, reports=reports, title=title, trace_url=trace_url)
    except Exception:
        logger.exception("failed to render nsys report")

    if attach and not clustered:
        try:
            await capture_report_file(report_path)
        except Exception:
            logger.exception("failed to attach .nsys-rep as a trace output")

    return summary


async def _persist_trace(report_path: str) -> Optional[str]:
    """Upload the .nsys-rep to durable storage (controller-free) and return its remote path.

    Used on clustered/jobset workers, which have no controller for the @flyte.trace File output.
    File.from_local uploads to the task's raw-data path, so the trace outlives the pod and the report
    can link to it.
    """
    f = await File.from_local(report_path)
    logger.info("nsys: saved .nsys-rep to %s", f.path)
    return f.path


def finalize_sync(
    report_path: str,
    *,
    title: Optional[str] = None,
    reports: Sequence[str] = _report.DEFAULT_REPORTS,
    attach: bool = True,
) -> dict:
    """Blocking twin of finalize, for a `with nsys.range(...)` block in a non-async task body.

    Mirrors finalize exactly, using the sync-native flyte primitives (File.from_local_sync, the sync
    report flush, the sync @flyte.trace path) so a non-async task summarizes and attaches its trace
    without an event loop. Best-effort throughout: a task must not fail because its profile could not
    be summarized.
    """
    if not _exists(report_path):
        logger.warning("nsys report not found at %s; skipping summary", report_path)
        return {}

    clustered = bool(os.environ.get("TORCHELASTIC_RUN_ID"))
    trace_url: Optional[str] = None
    if attach and clustered:
        try:
            trace_url = _persist_trace_sync(report_path)
        except Exception:
            logger.exception("nsys: failed to persist .nsys-rep for clustered worker")

    summary: dict = {}
    try:
        summary = _report.render_sync(report_path, reports=reports, title=title, trace_url=trace_url)
    except Exception:
        logger.exception("failed to render nsys report")

    if attach and not clustered:
        try:
            capture_report_file_sync(report_path)
        except Exception:
            logger.exception("failed to attach .nsys-rep as a trace output")

    return summary


def _persist_trace_sync(report_path: str) -> Optional[str]:
    """Blocking twin of _persist_trace: upload the .nsys-rep to durable storage for a controller-free worker."""
    f = File.from_local_sync(report_path)
    logger.info("nsys: saved .nsys-rep to %s", f.path)
    return f.path
