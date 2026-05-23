# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
#    "flyte",
#    "flyteplugins-union",
# ]
#
# [tool.uv.sources]
# flyteplugins-union = { path = "../../../../unionai/flyteplugins-union", editable = true }
# ///
"""
Volume benchmark driver.

Sweeps ``workload * engine * writeback`` and renders a multi-tab Flyte
report with timing tables and inline-SVG bar charts. Each cell runs in
its own sub-action with a fresh Volume; the driver fans the cells out
with ``asyncio.gather`` so the matrix runs concurrently.

Workloads (default set):

* ``metadata_burst`` — touch N empty files; stresses metadata-write rate.
* ``big_files`` — write K large blobs; stresses chunk-upload throughput.
* ``small_files`` — write K small files with payload; mixed metadata+IO.
* ``fork_burst`` — fork the base K times back-to-back; metadata snapshot
  + upload cost only.

Override the sweep at runtime, e.g.::

    uv run flyte run examples/volumes/bench.py main \\
        --workloads '["metadata_burst", "fork_burst"]'
"""

import asyncio
import logging
import math
import os
import time
import uuid
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from flyteplugins.union.io.volume import Volume, with_volume_deps

import flyte
import flyte.report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("bench")

IDL2 = "git+https://github.com/flyteorg/flyte.git@v2#subdirectory=gen/python"
_base = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-bench")
    .with_apt_packages("git")
    .with_pip_packages(IDL2, "kubernetes")
)
image = with_volume_deps(_base, install_flyte=False, install_local=True).with_local_v2()

env = flyte.TaskEnvironment(
    name="vol-bench",
    enable_fuse_mount=True,
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------
# Each workload runs against an already-mounted Volume rooted at /workspace
# and returns a dict of measurements. Keep them deterministic so cells are
# comparable across cells.

WorkloadFn = Callable[..., Awaitable[Dict[str, float]]]


async def _metadata_burst(_vol: Volume, *, n: int) -> Dict[str, float]:
    data = Path("/workspace/data")
    data.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for i in range(n):
        (data / f"f{i:08d}").touch()
    return {"workload_ms": (time.monotonic() - t0) * 1000.0, "items": float(n)}


async def _big_files(_vol: Volume, *, k: int, size_mib: float) -> Dict[str, float]:
    nbytes = int(size_mib * 1024 * 1024)
    payload = b"\0" * nbytes
    t0 = time.monotonic()
    for i in range(k):
        Path(f"/workspace/blob_{i:03d}.bin").write_bytes(payload)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    return {
        "workload_ms": elapsed_ms,
        "items": float(k),
        "throughput_mibs": (k * size_mib) / max(elapsed_ms / 1000.0, 1e-6),
    }


async def _small_files(_vol: Volume, *, k: int, size_bytes: int) -> Dict[str, float]:
    payload = b"x" * size_bytes
    data = Path("/workspace/small")
    data.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for i in range(k):
        (data / f"s{i:06d}").write_bytes(payload)
    return {"workload_ms": (time.monotonic() - t0) * 1000.0, "items": float(k)}


async def _fork_burst(vol: Volume, *, k: int, base_files: int = 100) -> Dict[str, float]:
    data = Path("/workspace/data")
    data.mkdir(parents=True, exist_ok=True)
    for i in range(base_files):
        (data / f"base_{i:04d}").write_text("x")

    fork_ms: List[float] = []
    for i in range(k):
        t0 = time.monotonic()
        await vol.fork(name=f"{vol.name}-fork-{i:04d}")
        fork_ms.append((time.monotonic() - t0) * 1000.0)

    p99 = quantiles(fork_ms, n=100, method="inclusive")[98] if len(fork_ms) >= 2 else fork_ms[0]
    return {
        "workload_ms": sum(fork_ms),
        "items": float(k),
        "fork_mean": mean(fork_ms),
        "fork_p50": median(fork_ms),
        "fork_p99": p99,
    }


# Self-managed workloads handle their own Volume lifecycle (no cell-level
# mount/commit). Used by cold_fork, which deliberately forks an *unmounted*
# Volume to exercise Volume.fork()'s cold path.
async def _cold_fork(
    *,
    engine: str,
    writeback: bool,
    k: int,
    base_files: int = 100,
    base_bytes: int = 1024,
) -> Tuple[Volume, Dict[str, float]]:
    suffix = uuid.uuid4().hex[:6]
    parent = Volume.empty(name=f"bench-cold-fork-{engine}-{suffix}", metadata_store_type=engine)

    t0 = time.monotonic()
    await parent.mount(writeback=writeback)
    mount_ms = (time.monotonic() - t0) * 1000.0

    data = Path("/workspace/data")
    data.mkdir(parents=True, exist_ok=True)
    payload = "x" * base_bytes
    for i in range(base_files):
        (data / f"base_{i:04d}").write_text(payload)

    t0 = time.monotonic()
    committed = await parent.commit()
    commit_ms = (time.monotonic() - t0) * 1000.0

    # Forks are issued on the *committed* (unmounted) Volume → cold path.
    fork_ms: List[float] = []
    for i in range(k):
        t0 = time.monotonic()
        await committed.fork(name=f"{committed.name}-cold-{i:04d}")
        fork_ms.append((time.monotonic() - t0) * 1000.0)

    p99 = quantiles(fork_ms, n=100, method="inclusive")[98] if len(fork_ms) >= 2 else fork_ms[0]
    return committed, {
        "workload_ms": sum(fork_ms),
        "items": float(k),
        "fork_mean": mean(fork_ms),
        "fork_p50": median(fork_ms),
        "fork_p99": p99,
        "mount_ms": mount_ms,
        "commit_ms": commit_ms,
        "index_bytes": 0.0,
        "used_bytes": float(committed.used_bytes) if committed.used_bytes is not None else 0.0,
        "inode_count": float(committed.inode_count) if committed.inode_count is not None else 0.0,
    }


WORKLOADS: Dict[str, Tuple[WorkloadFn, Dict[str, object]]] = {
    "metadata_burst": (_metadata_burst, {"n": 10_000}),
    "big_files": (_big_files, {"k": 5, "size_mib": 100}),
    "small_files": (_small_files, {"k": 5_000, "size_bytes": 4096}),
    "fork_burst": (_fork_burst, {"k": 25}),
}

# Workloads that manage their own Volume (skip the cell-level mount/commit).
SELF_MANAGED: Dict[str, Tuple[Callable[..., Awaitable[Dict[str, float]]], Dict[str, object]]] = {
    "cold_fork": (_cold_fork, {"k": 25, "base_files": 100, "base_bytes": 1024}),
}

ENGINES: List[str] = ["redis", "sqlite", "badger"]
WRITEBACK: List[bool] = [True, False]

# Workloads that exercise Volume.fork(). Badger's fork-counter bump path
# disagrees with the juicefs dump key casing today (separate fix), so
# fork workloads run against redis/sqlite only.
_FORK_WORKLOADS: set = {"fork_burst", "cold_fork"}
_FORK_ENGINES: List[str] = ["redis", "sqlite"]

# Dataset sweep: hold per-file size constant, vary count, plot one line per
# engine. Counts capped at 100k (small) / 10k (big) so a full sweep stays
# bounded in time and disk; bump in-place if you need to push further.
DATASET_SMALL_COUNTS: List[int] = [10, 100, 1_000, 10_000, 100_000]
DATASET_BIG_COUNTS: List[int] = [10, 100, 1_000, 10_000]
DATASET_SMALL_SIZE_BYTES = 256
DATASET_BIG_SIZE_KIB = 64
DATASET_ENGINES: List[str] = ["redis", "sqlite", "badger"]


# ---------------------------------------------------------------------------
# Per-cell sub-action
# ---------------------------------------------------------------------------


@env.task
async def run_cell(workload: str, engine: str, writeback: bool) -> Tuple[Volume, Dict[str, float]]:
    """Run one matrix cell.

    Returns ``(committed_volume, stats)`` so the Volume's serialized
    metadata (bucket, index path, parent, used_bytes, etc.) is visible
    in the Flyte UI alongside the timing numbers.
    """
    if workload in SELF_MANAGED:
        fn, params = SELF_MANAGED[workload]
        return await fn(engine=engine, writeback=writeback, **params)

    fn, params = WORKLOADS[workload]
    suffix = uuid.uuid4().hex[:6]
    # Volume names: alphanumerics and dashes only, 3-63 chars.
    safe_workload = workload.replace("_", "-")
    vol = Volume.empty(
        name=f"bench-{safe_workload}-{engine}-{int(writeback)}-{suffix}",
        metadata_store_type=engine,
    )

    t0 = time.monotonic()
    await vol.mount(writeback=writeback)
    mount_ms = (time.monotonic() - t0) * 1000.0

    result = await fn(vol, **params)

    t0 = time.monotonic()
    final = await vol.commit()
    commit_ms = (time.monotonic() - t0) * 1000.0

    # Capture the index footprint *after* commit. Redis only writes its
    # dump.rdb when SAVE runs (which commit() triggers); SQLite's index.db
    # is mutated continuously but commit() WAL-checkpoints it. Either way,
    # post-commit is the size of the file that actually gets uploaded.
    index_bytes = 0.0
    for p in ("/var/lib/flyte-volume/index.db", "/var/lib/flyte-volume/dump.rdb"):
        if os.path.exists(p):
            index_bytes = float(os.path.getsize(p))
            break

    # Volume-level stats populated by commit() (best-effort).
    used_bytes = float(final.used_bytes) if final.used_bytes is not None else 0.0
    inode_count = float(final.inode_count) if final.inode_count is not None else 0.0

    logger.info(
        "cell done: workload=%s engine=%s writeback=%s mount=%.0fms commit=%.0fms used=%.0f inodes=%.0f",
        workload,
        engine,
        writeback,
        mount_ms,
        commit_ms,
        used_bytes,
        inode_count,
    )

    stats = {
        **result,
        "mount_ms": mount_ms,
        "commit_ms": commit_ms,
        "index_bytes": index_bytes,
        "used_bytes": used_bytes,
        "inode_count": inode_count,
    }
    return final, stats


# ---------------------------------------------------------------------------
# Dataset-sweep cell
# ---------------------------------------------------------------------------


@env.task
async def run_dataset_cell(workload: str, engine: str, n: int) -> Tuple[Volume, Dict[str, float]]:
    """One ``(workload, engine, n)`` point on the dataset sweep.

    Holds per-file size constant (set by ``DATASET_SMALL_SIZE_BYTES`` /
    ``DATASET_BIG_SIZE_KIB``) and varies ``n``. Always runs with
    ``writeback=True`` so the curve isn't dominated by per-chunk upload
    latency. Returns ``(committed_volume, stats)`` so the Volume's
    metadata is inspectable in the Flyte UI.
    """
    suffix = uuid.uuid4().hex[:6]
    vol = Volume.empty(
        name=f"bench-ds-{workload.replace('_', '-')}-{engine}-{n}-{suffix}",
        metadata_store_type=engine,
    )

    t0 = time.monotonic()
    await vol.mount(writeback=True)
    mount_ms = (time.monotonic() - t0) * 1000.0

    if workload == "small_files":
        result = await _small_files(vol, k=n, size_bytes=DATASET_SMALL_SIZE_BYTES)
    elif workload == "big_files":
        result = await _big_files(vol, k=n, size_mib=DATASET_BIG_SIZE_KIB / 1024.0)
    else:
        raise ValueError(f"unsupported dataset workload: {workload}")

    t0 = time.monotonic()
    final = await vol.commit()
    commit_ms = (time.monotonic() - t0) * 1000.0

    return final, {**result, "mount_ms": mount_ms, "commit_ms": commit_ms}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell_label(engine: str, writeback: bool) -> str:
    return f"{engine}/{'wb' if writeback else 'cold'}"


def _bar_svg(title: str, labels: List[str], values: List[float], unit: str) -> str:
    w, h = 480, 220
    mL, mR, mT, mB = 60, 16, 32, 60
    pw, ph = w - mL - mR, h - mT - mB
    vmax = max([*values, 1.0])

    n = max(len(values), 1)
    slot = pw / n
    bar_w = slot * 0.7

    parts: List[str] = [
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        'style="max-width:100%;height:auto;font-family:system-ui,sans-serif">',
        f'<text x="{w / 2}" y="20" text-anchor="middle" font-size="13" font-weight="600">{title}</text>',
        f'<line x1="{mL}" y1="{mT + ph}" x2="{mL + pw}" y2="{mT + ph}" stroke="#888"/>',
        f'<line x1="{mL}" y1="{mT}" x2="{mL}" y2="{mT + ph}" stroke="#888"/>',
        f'<text x="{mL - 6}" y="{mT + 4}" text-anchor="end" font-size="10">{vmax:,.0f}</text>',
        f'<text x="{mL - 6}" y="{mT + ph + 4}" text-anchor="end" font-size="10">0</text>',
        f'<text x="14" y="{mT + ph / 2}" font-size="10" '
        f'transform="rotate(-90 14 {mT + ph / 2})" text-anchor="middle">{unit}</text>',
    ]

    # Lighter shades for the "cold" (writeback=off) variant so wb vs cold
    # reads at a glance within an engine.
    cold_shades = {"#1f6feb": "#94c7ff", "#fb8500": "#ffd6a3", "#0a8754": "#9ddfb8"}
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = mL + i * slot + (slot - bar_w) / 2
        bh = (v / vmax) * ph if vmax > 0 else 0.0
        y = mT + ph - bh
        engine = lab.split("/", 1)[0]
        color = _ENGINE_COLORS.get(engine, "#666")
        if lab.endswith("cold"):
            color = cold_shades.get(color, color)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="10">{v:,.0f}</text>'
        )
        # Two-line x label (engine / mode), rotated for readability.
        cx = x + bar_w / 2
        parts.append(f'<text x="{cx:.1f}" y="{mT + ph + 14:.1f}" text-anchor="middle" font-size="10">{lab}</text>')

    parts.append("</svg>")
    return "".join(parts)


_ENGINE_COLORS = {"redis": "#1f6feb", "sqlite": "#fb8500", "badger": "#0a8754"}


def _line_svg(
    title: str,
    series: Dict[str, List[Tuple[float, float]]],
    x_label: str,
    y_label: str,
    *,
    log_x: bool = True,
    log_y: bool = True,
) -> str:
    """Log-log line plot with one series per name. ``series`` is ``{name: [(x, y), ...]}``.

    Lifts log-scale plotting into pure SVG so the report stays self-contained
    (no JS, no Plotly). One polyline + circles per series; right-side legend.
    """
    w, h = 560, 320
    mL, mR, mT, mB = 70, 110, 36, 56
    pw, ph = w - mL - mR, h - mT - mB

    all_pts = [(x, y) for pts in series.values() for x, y in pts if x > 0 and y > 0]
    if not all_pts:
        return (
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:100%;height:auto;font-family:system-ui,sans-serif">'
            f'<text x="{w / 2}" y="{h / 2}" text-anchor="middle" font-size="13">no data</text></svg>'
        )

    def tx(v: float) -> float:
        return math.log10(v) if log_x else v

    def ty(v: float) -> float:
        return math.log10(v) if log_y else v

    xs = [tx(x) for x, _ in all_pts]
    ys = [ty(y) for _, y in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if log_x:
        x_min, x_max = math.floor(x_min), math.ceil(x_max)
    if log_y:
        y_min, y_max = math.floor(y_min), math.ceil(y_max)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1

    def px(v: float) -> float:
        return mL + (tx(v) - x_min) / (x_max - x_min) * pw

    def py(v: float) -> float:
        return mT + ph - (ty(v) - y_min) / (y_max - y_min) * ph

    parts: List[str] = [
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        'style="max-width:100%;height:auto;font-family:system-ui,sans-serif">',
        f'<text x="{w / 2}" y="20" text-anchor="middle" font-size="14" font-weight="600">{title}</text>',
        f'<line x1="{mL}" y1="{mT + ph}" x2="{mL + pw}" y2="{mT + ph}" stroke="#666"/>',
        f'<line x1="{mL}" y1="{mT}" x2="{mL}" y2="{mT + ph}" stroke="#666"/>',
    ]

    # Decade gridlines + tick labels.
    if log_x:
        for e in range(int(x_min), int(x_max) + 1):
            v = 10**e
            x = px(v)
            parts.append(f'<line x1="{x:.1f}" y1="{mT}" x2="{x:.1f}" y2="{mT + ph}" stroke="#eee"/>')
            parts.append(f'<text x="{x:.1f}" y="{mT + ph + 14}" text-anchor="middle" font-size="10">{int(v):,}</text>')
    if log_y:
        for e in range(int(y_min), int(y_max) + 1):
            v = 10**e
            y = py(v)
            parts.append(f'<line x1="{mL}" y1="{y:.1f}" x2="{mL + pw}" y2="{y:.1f}" stroke="#eee"/>')
            label = f"{int(v):,}" if v >= 1 else f"{v:.2g}"
            parts.append(f'<text x="{mL - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10">{label}</text>')

    parts.append(
        f'<text x="{mL + pw / 2}" y="{h - 12}" text-anchor="middle" font-size="11">{x_label}</text>'
    )
    parts.append(
        f'<text x="14" y="{mT + ph / 2}" text-anchor="middle" font-size="11" '
        f'transform="rotate(-90 14 {mT + ph / 2})">{y_label}</text>'
    )

    legend_y = mT
    for name, points in series.items():
        color = _ENGINE_COLORS.get(name, "#666")
        pts = sorted(p for p in points if p[0] > 0 and p[1] > 0)
        if pts:
            polyline = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
            parts.append(
                f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for x, y in pts:
                parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}"/>')
        parts.append(f'<rect x="{mL + pw + 12}" y="{legend_y}" width="14" height="3" fill="{color}"/>')
        parts.append(f'<text x="{mL + pw + 30}" y="{legend_y + 5}" font-size="11">{name}</text>')
        legend_y += 18

    parts.append("</svg>")
    return "".join(parts)


def _render_workload(name: str, rows: List[Dict[str, object]]) -> str:
    engine_order = {e: i for i, e in enumerate(ENGINES)}

    def sort_key(r: Dict[str, object]) -> Tuple[int, int]:
        return (engine_order.get(str(r["engine"]), 99), 0 if r["writeback"] else 1)

    rows = sorted(rows, key=sort_key)
    if not rows:
        return f"<h2>{name}</h2><p><em>no results</em></p>"

    base_cols = [
        "engine",
        "writeback",
        "items",
        "workload_ms",
        "mount_ms",
        "commit_ms",
        "index_bytes",
        "used_bytes",
        "inode_count",
    ]
    extra_cols = [c for c in ("throughput_mibs", "fork_mean", "fork_p50", "fork_p99") if c in rows[0]]
    cols = base_cols + extra_cols

    head = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, bool):
                cells.append(f"<td>{'on' if v else 'off'}</td>")
            elif isinstance(v, float):
                cells.append(f"<td align=right>{v:,.1f}</td>")
            elif isinstance(v, int):
                cells.append(f"<td align=right>{v:,}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        body += "<tr>" + "".join(cells) + "</tr>"
    table = f"<table border=1 cellpadding=6 cellspacing=0>{head}{body}</table>"

    labels = [_cell_label(str(r["engine"]), bool(r["writeback"])) for r in rows]
    charts = [
        _bar_svg("workload time", labels, [float(r["workload_ms"]) for r in rows], "ms"),
        _bar_svg("commit time", labels, [float(r["commit_ms"]) for r in rows], "ms"),
        _bar_svg("index size", labels, [float(r["index_bytes"]) for r in rows], "bytes"),
    ]
    if "throughput_mibs" in rows[0]:
        charts.append(_bar_svg("throughput", labels, [float(r["throughput_mibs"]) for r in rows], "MiB/s"))
    if "fork_mean" in rows[0]:
        charts.append(_bar_svg("fork mean", labels, [float(r["fork_mean"]) for r in rows], "ms"))

    chart_html = "".join(f'<div style="flex:1 1 320px;min-width:320px">{c}</div>' for c in charts)
    return f'<h2>{name}</h2>{table}<div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:12px">{chart_html}</div>'


def _render_overview(rows: List[Dict[str, object]], workloads: List[str]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{r['workload']}</td>"
        f"<td>{r['engine']}</td>"
        f"<td>{'on' if r['writeback'] else 'off'}</td>"
        f"<td align=right>{float(r['workload_ms']):,.0f}</td>"
        f"<td align=right>{float(r['mount_ms']):,.0f}</td>"
        f"<td align=right>{float(r['commit_ms']):,.0f}</td>"
        f"<td align=right>{int(r['index_bytes']):,}</td>"
        f"<td align=right>{int(float(r.get('used_bytes', 0))):,}</td>"
        f"<td align=right>{int(float(r.get('inode_count', 0))):,}</td>"
        "</tr>"
        for w in workloads
        for r in (x for x in rows if x["workload"] == w)
    )
    head = (
        "<tr><th>workload</th><th>engine</th><th>writeback</th>"
        "<th>workload_ms</th><th>mount_ms</th><th>commit_ms</th>"
        "<th>index_bytes</th><th>used_bytes</th><th>inodes</th></tr>"
    )
    return (
        "<h1>Volume benchmark sweep</h1>"
        "<p>Each row is one sub-action with a fresh <code>Volume</code>. "
        "Sub-actions fan out concurrently via <code>asyncio.gather</code>.</p>"
        f"<table border=1 cellpadding=6 cellspacing=0>{head}{body}</table>"
    )


def _render_dataset_sweep(
    small_rows: List[Dict[str, object]],
    big_rows: List[Dict[str, object]],
) -> str:
    """Two side-by-side line plots: small_files vs big_files, one line per engine."""

    def _by_engine(rows: List[Dict[str, object]]) -> Dict[str, List[Tuple[float, float]]]:
        by: Dict[str, List[Tuple[float, float]]] = {}
        for r in rows:
            by.setdefault(str(r["engine"]), []).append((float(r["n"]), float(r["workload_ms"])))
        return by

    small_chart = _line_svg(
        title=f"small_files — {DATASET_SMALL_SIZE_BYTES} B per file",
        series=_by_engine(small_rows),
        x_label="file count",
        y_label="workload_ms",
    )
    big_chart = _line_svg(
        title=f"big_files — {DATASET_BIG_SIZE_KIB} KiB per file",
        series=_by_engine(big_rows),
        x_label="file count",
        y_label="workload_ms",
    )

    def _table(rows: List[Dict[str, object]]) -> str:
        rows = sorted(rows, key=lambda r: (str(r["engine"]), int(r["n"])))
        head = (
            "<tr><th>engine</th><th>n</th>"
            "<th>workload_ms</th><th>mount_ms</th><th>commit_ms</th></tr>"
        )
        body = "".join(
            "<tr>"
            f"<td>{r['engine']}</td>"
            f"<td align=right>{int(r['n']):,}</td>"
            f"<td align=right>{float(r['workload_ms']):,.0f}</td>"
            f"<td align=right>{float(r['mount_ms']):,.0f}</td>"
            f"<td align=right>{float(r['commit_ms']):,.0f}</td>"
            "</tr>"
            for r in rows
        )
        return f"<table border=1 cellpadding=6 cellspacing=0>{head}{body}</table>"

    return (
        "<h1>Dataset size sweep</h1>"
        "<p>Each cell is a fresh Volume with writeback enabled. "
        "X-axis is the file count; y-axis is the workload time (ms). "
        "Lines are metadata engines; both axes are log-scale.</p>"
        '<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start">'
        f'<div style="flex:1 1 480px;min-width:480px">{small_chart}{_table(small_rows)}</div>'
        f'<div style="flex:1 1 480px;min-width:480px">{big_chart}{_table(big_rows)}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@env.task(report=True)
async def volume_benchmark_driver(
    workloads: Optional[List[str]] = None,
    engines: Optional[List[str]] = None,
    writeback: Optional[List[bool]] = None,
    dataset_sweep: bool = False,
) -> str:
    all_workloads = {**WORKLOADS, **SELF_MANAGED}
    selected_workloads = workloads or list(all_workloads.keys())
    selected_engines = engines or ENGINES
    selected_writeback = writeback if writeback is not None else WRITEBACK

    unknown = [w for w in selected_workloads if w not in all_workloads]
    if unknown:
        raise ValueError(f"unknown workloads: {unknown}; available: {list(all_workloads)}")

    keys: List[Tuple[str, str, bool]] = []
    coros = []
    for wname in selected_workloads:
        engines_for_workload = (
            [e for e in selected_engines if e in _FORK_ENGINES] if wname in _FORK_WORKLOADS else selected_engines
        )
        for engine in engines_for_workload:
            for wb in selected_writeback:
                keys.append((wname, engine, wb))
                short = f"{wname.replace('_', '-')}-{engine}-{'wb' if wb else 'cold'}"
                coros.append(run_cell.override(short_name=short)(workload=wname, engine=engine, writeback=wb))

    # Dataset-size sweep cells: (workload, engine, n). Same per-file size
    # across a workload, varying count; one curve per engine.
    ds_keys: List[Tuple[str, str, int]] = []
    ds_coros = []
    if dataset_sweep:
        for engine in DATASET_ENGINES:
            for n in DATASET_SMALL_COUNTS:
                ds_keys.append(("small_files", engine, n))
                ds_coros.append(
                    run_dataset_cell.override(short_name=f"ds-small-{engine}-{n}")(
                        workload="small_files", engine=engine, n=n
                    )
                )
            for n in DATASET_BIG_COUNTS:
                ds_keys.append(("big_files", engine, n))
                ds_coros.append(
                    run_dataset_cell.override(short_name=f"ds-big-{engine}-{n}")(
                        workload="big_files", engine=engine, n=n
                    )
                )

    logger.info(
        "dispatching %d matrix cells + %d dataset cells across %d workloads",
        len(coros),
        len(ds_coros),
        len(selected_workloads),
    )
    raw = await asyncio.gather(*coros, *ds_coros)
    matrix_results = raw[: len(coros)]
    ds_results = raw[len(coros) :]

    rows: List[Dict[str, object]] = []
    for (wname, engine, wb), (_vol, stats) in zip(keys, matrix_results):
        rows.append({"workload": wname, "engine": engine, "writeback": wb, **stats})

    ds_rows: List[Dict[str, object]] = []
    for (wname, engine, n), (_vol, stats) in zip(ds_keys, ds_results):
        ds_rows.append({"workload": wname, "engine": engine, "n": n, **stats})

    overview_tab = flyte.report.get_tab("Overview")
    overview_tab.log(_render_overview(rows, selected_workloads))

    for wname in selected_workloads:
        wrows = [r for r in rows if r["workload"] == wname]
        tab = flyte.report.get_tab(wname)
        tab.log(_render_workload(wname, wrows))

    if dataset_sweep:
        small_rows = [r for r in ds_rows if r["workload"] == "small_files"]
        big_rows = [r for r in ds_rows if r["workload"] == "big_files"]
        sweep_tab = flyte.report.get_tab("Dataset sweep")
        sweep_tab.log(_render_dataset_sweep(small_rows, big_rows))

    await flyte.report.flush.aio()
    return (
        f"{len(rows)} matrix cells + {len(ds_rows)} dataset cells "
        f"across {len(selected_workloads)} workloads"
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(volume_benchmark_driver, dataset_sweep=True)
    print(run.url)
    run.wait()
