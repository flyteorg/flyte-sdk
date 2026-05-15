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
Volume benchmark: ``mount()`` + ``commit()`` overhead vs. index size.

For each N in --sizes, this workflow:

1. ``populate`` — boots a fresh Volume and creates N empty files under
   ``/workspace/data/``, then commits.
2. ``measure`` — downstream task that mounts the committed Volume,
   records the time it took, reads the on-disk index size, then commits
   and records that time.

Results are rendered as an HTML table in the main task's Report.

Empty-file creation isolates metadata cost (no S3 chunk uploads), so the
numbers track index growth, not bandwidth. Expect roughly linear growth
in index size and a smaller-than-linear increase in mount/commit time
(both download and upload are bulk transfers of the SQLite file).
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import flyte
import flyte.report
from flyte.extras import Volume, volume_image, volume_pod_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("bench-mc")

IDL2 = "git+https://github.com/flyteorg/flyte.git@v2#subdirectory=gen/python"
_base_image = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-bench-mc")
    .with_apt_packages("git")
    .with_pip_packages(IDL2, "kubernetes")
)
image = volume_image(_base_image).with_local_v2()

env = flyte.TaskEnvironment(
    name="vol-bench-mc",
    pod_template=volume_pod_template(),
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


@env.task
async def populate(name: str, n_files: int) -> Volume:
    """Bootstrap a fresh volume containing *n_files* empty files."""
    logger.info("populate: name=%s n_files=%d", name, n_files)
    vol = Volume.empty(name=name)
    await vol.mount()
    data = Path("/workspace/data")
    data.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for i in range(n_files):
        (data / f"f{i:08d}").touch()
    elapsed = time.monotonic() - t0
    logger.info(
        "populate: created %d files in %.2fs (%.2f ms/file)",
        n_files, elapsed, (elapsed * 1000.0) / max(n_files, 1),
    )
    return await vol.commit()


@env.task
async def measure(vol: Volume) -> Dict[str, float]:
    """Time mount() and commit() for an already-populated volume."""
    t0 = time.monotonic()
    await vol.mount()
    mount_ms = (time.monotonic() - t0) * 1000.0

    # Index file lives under /var/lib/flyte-volume; name depends on engine.
    index_bytes: float = -1.0
    for candidate in ("/var/lib/flyte-volume/index.db", "/var/lib/flyte-volume/dump.rdb"):
        try:
            index_bytes = float(os.path.getsize(candidate))
            break
        except OSError:
            continue

    t1 = time.monotonic()
    _ = await vol.commit()
    commit_ms = (time.monotonic() - t1) * 1000.0

    logger.info(
        "measure: index=%d bytes mount=%.1fms commit=%.1fms",
        int(index_bytes), mount_ms, commit_ms,
    )
    return {"index_bytes": index_bytes, "mount_ms": mount_ms, "commit_ms": commit_ms}


def _render(rows: List[Dict[str, float]]) -> str:
    head = (
        "<tr>"
        "<th>files</th><th>index_bytes</th><th>index_MiB</th>"
        "<th>mount_ms</th><th>commit_ms</th>"
        "</tr>"
    )
    body = "".join(
        f"<tr>"
        f"<td align=right>{int(r['n_files']):,}</td>"
        f"<td align=right>{int(r['index_bytes']):,}</td>"
        f"<td align=right>{r['index_bytes'] / (1 << 20):.2f}</td>"
        f"<td align=right>{r['mount_ms']:.1f}</td>"
        f"<td align=right>{r['commit_ms']:.1f}</td>"
        f"</tr>"
        for r in rows
    )
    return (
        "<h2>Volume mount/commit overhead by index size</h2>"
        "<table border=1 cellpadding=6 cellspacing=0>"
        f"{head}{body}"
        "</table>"
    )


@env.task(report=True)
async def main(sizes: Optional[List[int]] = None) -> str:
    if sizes is None:
        sizes = [1_000, 10_000, 100_000]
    run_id = uuid.uuid4().hex[:8]

    rows: List[Dict[str, float]] = []
    for n in sizes:
        vol = await populate(name=f"bench-mc-{run_id}-{n}", n_files=n)
        m = await measure(vol)
        rows.append({"n_files": float(n), **m})

    html = _render(rows)
    tab = flyte.report.get_tab("Results")
    tab.log(html)
    await flyte.report.flush.aio()
    return html


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
