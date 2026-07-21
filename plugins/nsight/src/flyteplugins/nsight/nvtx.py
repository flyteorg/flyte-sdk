"""NVTX annotation helpers.

Labels regions of your code so they show up as named spans on the Nsight timeline and in the
NVTX summary of the report. Thin wrappers over torch.cuda.nvtx so you annotate without importing
torch internals, and a no-op when torch or CUDA is unavailable, so the same code runs unchanged
off-GPU and outside a profiling run.

    from flyteplugins.nsight import nvtx

    with nvtx.range("forward"):
        out = model(x)

    nvtx.mark("checkpoint saved")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


def _nvtx():
    try:
        import torch.cuda.nvtx as _n

        return _n
    except Exception:  # pragma: no cover - torch not installed / no CUDA
        return None


@contextmanager
def range(message: str) -> Iterator[None]:
    """Push an NVTX range on enter and pop it on exit. No-op if NVTX is unavailable."""
    n = _nvtx()
    if n is None:
        yield
        return
    n.range_push(message)
    try:
        yield
    finally:
        n.range_pop()


def mark(message: str) -> None:
    """Drop a single NVTX marker at this instant. No-op if NVTX is unavailable."""
    n = _nvtx()
    if n is not None:
        n.mark(message)
