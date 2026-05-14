# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
#    "flyte",
# ]
#
# [tool.uv.sources]
# flyte = { path = "../..", editable = true }
# ///
"""
Volume benchmark: fork cost + isolation.

A ``Volume.fork()`` is metadata-only: it ``BACKUP``s the SQLite index in
place and uploads the snapshot. Chunks in object storage are shared, so
fork time should track index size, not data volume.

This workflow:

1. ``populate_base`` — bootstraps a base volume containing ``N_BASE_FILES``
   small files plus a single ``BLOB_BYTES`` blob, so the chunk store is
   non-trivial.
2. ``make_forks(base, k)`` — mounts the base **once** and calls
   ``vol.fork()`` *k* times in a tight loop, recording per-fork wall time.
   This isolates fork() itself from any mount/commit overhead.
3. ``verify_isolation`` — picks a sample of forks, writes a unique marker
   to each, then re-mounts each and checks that
     a) the marker survived,
     b) the base's files are still visible,
     c) sibling markers are *not* visible (true namespace isolation).
4. Renders timing percentiles + the isolation verdict as an HTML report.

Use ``--k 10`` to sanity-check the flow, ``--k 100`` or ``--k 1000`` to
stress.
"""

import logging
import random
import time
import uuid
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Dict, List, Tuple

import flyte
import flyte.report
from flyte.extras import Volume, volume_image, volume_pod_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("bench-fork")

IDL2 = "git+https://github.com/flyteorg/flyte.git@v2#subdirectory=gen/python"
_base_image = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-bench-fork")
    .with_apt_packages("git")
    .with_pip_packages(IDL2, "kubernetes")
)
image = volume_image(_base_image).with_local_v2()

env = flyte.TaskEnvironment(
    name="vol-bench-fork",
    pod_template=volume_pod_template(),
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
)

N_BASE_FILES = 1_000
BLOB_BYTES = 100 * 1024 * 1024  # 100 MiB


@env.task
async def populate_base(name: str) -> Volume:
    logger.info("populate_base: name=%s", name)
    vol = Volume.empty(name=name)
    await vol.mount()
    data = Path("/workspace/data")
    data.mkdir(parents=True, exist_ok=True)
    for i in range(N_BASE_FILES):
        (data / f"base_{i:05d}.txt").write_text(f"file {i}\n")
    Path("/workspace/blob.bin").write_bytes(b"\0" * BLOB_BYTES)
    return await vol.commit()


@env.task
async def make_forks(base: Volume, k: int) -> Tuple[List[Volume], List[float]]:
    """Mount base once, then call fork() k times back-to-back. Returns
    (forks, per-fork timings in ms)."""
    logger.info("make_forks: k=%d", k)
    await base.mount()

    fork_ms: List[float] = []
    forks: List[Volume] = []
    for i in range(k):
        t0 = time.monotonic()
        f = await base.fork(name=f"{base.name}-fork-{i:05d}")
        fork_ms.append((time.monotonic() - t0) * 1000.0)
        forks.append(f)
        if (i + 1) % 100 == 0:
            logger.info("make_forks: %d/%d forks created", i + 1, k)

    return forks, fork_ms


@env.task
async def verify_isolation(base: Volume, forks: List[Volume], sample_size: int) -> Dict[str, object]:
    """Write unique markers to a sample of forks, then re-mount each and
    check that each fork sees only its own marker plus the base content.
    """
    rng = random.Random(0)
    sample = rng.sample(forks, min(sample_size, len(forks)))
    logger.info("verify_isolation: sampling %d forks", len(sample))

    # Pass 1: write a marker into each sampled fork.
    written: List[Volume] = []
    for f in sample:
        await f.mount()
        Path(f"/workspace/marker__{f.name}.txt").write_text(f"hello from {f.name}\n")
        written.append(await f.commit())

    # Pass 2: re-mount each fork and check what's visible.
    issues: List[str] = []
    base_probe = f"/workspace/data/base_{N_BASE_FILES // 2:05d}.txt"
    for f in written:
        await f.mount()
        own_marker = Path(f"/workspace/marker__{f.name}.txt")
        if not own_marker.exists():
            issues.append(f"{f.name}: own marker missing")
        if not Path(base_probe).exists():
            issues.append(f"{f.name}: base file {base_probe} missing")
        if not Path("/workspace/blob.bin").exists():
            issues.append(f"{f.name}: base blob missing")
        for sibling in written:
            if sibling.name == f.name:
                continue
            sib_marker = Path(f"/workspace/marker__{sibling.name}.txt")
            if sib_marker.exists():
                issues.append(f"{f.name}: leaked marker from {sibling.name}")
        await f.commit()

    # Also confirm base is unchanged: no markers in it.
    await base.mount()
    base_leaks = [p.name for p in Path("/workspace").glob("marker__*.txt")]
    await base.commit()

    return {
        "checked": len(written),
        "issues": issues,
        "base_leaks": base_leaks,
    }


def _percentiles(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    qs = quantiles(samples, n=100, method="inclusive") if len(samples) > 1 else samples * 99
    return {
        "n": float(len(samples)),
        "mean": mean(samples),
        "p50": median(samples),
        "p90": qs[89] if len(qs) >= 90 else samples[-1],
        "p99": qs[98] if len(qs) >= 99 else samples[-1],
        "max": max(samples),
    }


def _render(k: int, stats: Dict[str, float], verdict: Dict[str, object]) -> str:
    timing_rows = "".join(
        f"<tr><td>{key}</td><td align=right>{stats[key]:.2f}</td></tr>"
        for key in ("n", "mean", "p50", "p90", "p99", "max")
    )
    issues = verdict.get("issues") or []
    base_leaks = verdict.get("base_leaks") or []
    isolation_ok = not issues and not base_leaks
    badge_color = "#1a7f37" if isolation_ok else "#cf222e"
    badge_text = "PASS" if isolation_ok else "FAIL"

    issue_html = ""
    if issues:
        issue_html += "<h4>Per-fork issues</h4><ul>" + "".join(
            f"<li><code>{i}</code></li>" for i in issues[:20]
        ) + "</ul>"
        if len(issues) > 20:
            issue_html += f"<p>... and {len(issues) - 20} more</p>"
    if base_leaks:
        issue_html += "<h4>Base contamination</h4><ul>" + "".join(
            f"<li><code>{leak}</code></li>" for leak in base_leaks
        ) + "</ul>"

    return f"""
        <h2>Volume fork benchmark — k = {k}</h2>
        <h3>Per-fork timing (ms)</h3>
        <table border=1 cellpadding=6 cellspacing=0>
          <tr><th>stat</th><th>value</th></tr>
          {timing_rows}
        </table>
        <h3>Isolation verdict
          <span style="background:{badge_color};color:white;padding:2px 8px;border-radius:4px;">
            {badge_text}
          </span>
        </h3>
        <p>{verdict['checked']} fork(s) checked.</p>
        {issue_html}
    """


@env.task(report=True)
async def main(k: int = 10) -> str:
    run_id = uuid.uuid4().hex[:8]
    base = await populate_base(name=f"bench-fork-{run_id}")
    forks, fork_ms = await make_forks(base, k=k)
    stats = _percentiles(fork_ms)

    sample_size = max(5, min(20, k // 50))  # 5 for small k, up to 20 for large
    verdict = await verify_isolation(base, forks, sample_size=sample_size)

    html = _render(k, stats, verdict)
    tab = flyte.report.get_tab("Results")
    tab.log(html)
    await flyte.report.flush.aio()
    return html


if __name__ == "__main__":
    flyte.init_from_config()
    # Adjust k via `flyte run`: --k 100 / --k 1000
    run = flyte.run(main, k=10)
    print(run.url)
    run.wait()
