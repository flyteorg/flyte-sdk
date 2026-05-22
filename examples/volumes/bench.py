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
import os
import time
import uuid
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import flyte
import flyte.report
from flyteplugins.union.io.volume import Volume, volume_image, volume_pod_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("bench")

IDL2 = "git+https://github.com/flyteorg/flyte.git@v2#subdirectory=gen/python"
_base = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-bench")
    .with_apt_packages("git")
    .with_pip_packages(IDL2, "kubernetes")
)
image = volume_image(_base, install_local=True)

env = flyte.TaskEnvironment(
    name="vol-bench",
    pod_template=volume_pod_template(),
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


async def _big_files(_vol: Volume, *, k: int, size_mib: int) -> Dict[str, float]:
    payload = b"\0" * (size_mib * 1024 * 1024)
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
) -> Dict[str, float]:
    suffix = uuid.uuid4().hex[:6]
    parent = Volume.empty(name=f"bench-cold-fork-{engine}-{suffix}", metadata_engine=engine)

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
    return {
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

ENGINES: List[str] = ["redis", "sqlite"]
WRITEBACK: List[bool] = [True, False]


# ---------------------------------------------------------------------------
# Per-cell sub-action
# ---------------------------------------------------------------------------


@env.task
async def run_cell(workload: str, engine: str, writeback: bool) -> Dict[str, float]:
    if workload in SELF_MANAGED:
        fn, params = SELF_MANAGED[workload]
        return await fn(engine=engine, writeback=writeback, **params)

    fn, params = WORKLOADS[workload]
    suffix = uuid.uuid4().hex[:6]
    # Volume names: alphanumerics and dashes only, 3-63 chars.
    safe_workload = workload.replace("_", "-")
    vol = Volume.empty(
        name=f"bench-{safe_workload}-{engine}-{int(writeback)}-{suffix}",
        metadata_engine=engine,
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

    return {
        **result,
        "mount_ms": mount_ms,
        "commit_ms": commit_ms,
        "index_bytes": index_bytes,
        "used_bytes": used_bytes,
        "inode_count": inode_count,
    }


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

    for i, (lab, v) in enumerate(zip(labels, values)):
        x = mL + i * slot + (slot - bar_w) / 2
        bh = (v / vmax) * ph if vmax > 0 else 0.0
        y = mT + ph - bh
        color = "#1f6feb" if lab.startswith("redis") else "#fb8500"
        if lab.endswith("cold"):
            color = "#94c7ff" if color == "#1f6feb" else "#ffd6a3"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="10">{v:,.0f}</text>'
        )
        # Two-line x label (engine / mode), rotated for readability.
        cx = x + bar_w / 2
        parts.append(f'<text x="{cx:.1f}" y="{mT + ph + 14:.1f}" text-anchor="middle" font-size="10">{lab}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _render_workload(name: str, rows: List[Dict[str, object]]) -> str:
    def sort_key(r: Dict[str, object]) -> Tuple[int, int]:
        return (0 if r["engine"] == "redis" else 1, 0 if r["writeback"] else 1)

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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@env.task(report=True)
async def volume_benchmark_driver(
    workloads: Optional[List[str]] = None,
    engines: Optional[List[str]] = None,
    writeback: Optional[List[bool]] = None,
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
        for engine in selected_engines:
            for wb in selected_writeback:
                keys.append((wname, engine, wb))
                short = f"{wname.replace('_', '-')}-{engine}-{'wb' if wb else 'cold'}"
                coros.append(run_cell.override(short_name=short)(workload=wname, engine=engine, writeback=wb))

    logger.info("dispatching %d cells across %d workloads", len(coros), len(selected_workloads))
    raw = await asyncio.gather(*coros)

    rows: List[Dict[str, object]] = []
    for (wname, engine, wb), result in zip(keys, raw):
        rows.append({"workload": wname, "engine": engine, "writeback": wb, **result})

    overview_tab = flyte.report.get_tab("Overview")
    overview_tab.log(_render_overview(rows, selected_workloads))

    for wname in selected_workloads:
        wrows = [r for r in rows if r["workload"] == wname]
        tab = flyte.report.get_tab(wname)
        tab.log(_render_workload(wname, wrows))

    await flyte.report.flush.aio()
    return f"{len(rows)} cells across {len(selected_workloads)} workloads"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(volume_benchmark_driver)
    print(run.url)
    run.wait()
