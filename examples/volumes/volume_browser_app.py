# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte",
#    "flyteplugins-union>=0.4.0",
#    "fastapi",
#    "uvicorn",
# ]
#
# [tool.uv.sources]
# flyte = { path = "../..", editable = true }
# ///
"""
Volume-backed file browser app — no FUSE, no privileges.

A task snapshots a git repo (the flyte-sdk repo by default) into a
``Volume`` and seals it into an immutable ``ROVolume``. A scale-from-zero
FastAPI app then serves a file browser over that sealed volume.

**The app never FUSE-mounts.** Knative (which backs Flyte apps) silently
drops ``privileged: true``, rejects ``capabilities.add`` and — before
Serving 1.17 — hostPath volumes, so the kernel-mount path that tasks use
(``enable_fuse_mount=True``) is unavailable to app pods on most clusters.
Instead, this example prototypes a userspace alternative,
:func:`serve_volume_webdav`: the same juicefs engine ``mount()`` uses
(download index -> open read-only session -> stream chunks from the
bucket) exposed over localhost WebDAV instead of through ``/dev/fuse``.
Zero privileges, works on stock Knative. This is the prototype for a
future ``ROVolume.serve()`` in flyteplugins-union.

How the volume crosses the task -> app boundary: app parameters carry
strings/files/dirs, not Volumes, so ``snapshot_repo`` returns the sealed
volume **twice** — once as an ``ROVolume`` (for lineage in the UI) and
once as its JSON spec (``model_dump_json``; Volume is a pydantic model).
The app declares a ``RunOutput`` parameter with ``getter=(1,)`` to pick
up the JSON of the latest successful snapshot, and ``@env.on_startup``
rebuilds the ``ROVolume`` and starts the WebDAV sidecar process.

Scale-from-zero is the default app scaling (``Scaling(replicas=(0, 1))``);
this example sets it explicitly plus a short ``scaledown_after``. Each cold
start re-runs ``on_startup`` — one index download + a subprocess spawn, no
data copy. The repo's chunks stream from object storage on first read.

Usage::

    # one-shot: snapshot the repo, then deploy the browser
    uv run examples/volumes/volume_browser_app.py

    # or separately
    uv run flyte run examples/volumes/volume_browser_app.py snapshot_repo
    uv run flyte serve examples/volumes/volume_browser_app.py env

Prereqs:
- A bucket the cluster's service account can read/write.
- For the *snapshot task* only: cluster nodes expose ``/dev/fuse`` and pods
  can run privileged (``enable_fuse_mount=True``). The app needs nothing.
"""

import asyncio
import html
import logging
import mimetypes
import os
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from flyteplugins.union.io import ROVolume, Volume

import flyte
import flyte.app
from flyte.app import Parameter, RunOutput
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("volume-browser")

VOL_NAME = os.environ.get("VOL_NAME", "flyte-sdk-src")
REPO_TARBALL_URL = os.environ.get(
    "REPO_TARBALL_URL", "https://github.com/flyteorg/flyte-sdk/archive/refs/heads/main.tar.gz"
)

# A plain image + the released flyteplugins-union package is all Volumes need
# (juicefs is bundled in the PyPI platform wheels). fastapi/uvicorn are for the
# browser app; the snapshot task shares the image so there's one build.
# NOTE: pip packages first, .with_local_v2() last — the local-SDK wheel layer
# installs with --no-deps --reinstall, so it must come after the pip layer or
# flyteplugins-union's `flyte>=2.3,<2.5` dep would re-resolve flyte from PyPI
# and clobber the local dev SDK.
image = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-browser")
    .with_pip_packages("flyteplugins-union>=0.4.0", "fastapi", "uvicorn")
    .with_local_v2()
)

# ---------------------------------------------------------------------------
# Snapshot task: repo tarball -> sealed ROVolume
# ---------------------------------------------------------------------------

task_env = flyte.TaskEnvironment(
    name="repo-snapshot",
    # CAP_SYS_ADMIN + /dev/fuse come from this flag; no PodTemplate needed.
    # Only the *task* mounts via FUSE — the app below serves without it.
    enable_fuse_mount=True,
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


def _extract_stripped(tarball: str, dest: str) -> int:
    """Extract ``tarball`` into ``dest``, stripping the top-level directory
    (GitHub tarballs wrap everything in ``<repo>-<ref>/``). Returns the
    number of extracted members."""
    count = 0
    with tarfile.open(tarball, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue  # the top-level dir itself
            m.name = parts[1]
            members.append(m)
        tf.extractall(dest, members=members, filter="data")
        count = len(members)
    return count


@task_env.task
async def snapshot_repo(tarball_url: str = REPO_TARBALL_URL) -> Tuple[ROVolume, str]:
    """Download a repo tarball into a fresh Volume and seal it.

    Returns the sealed ``ROVolume`` (kept typed for lineage in the UI) and
    its JSON spec — the string the app's ``RunOutput`` parameter picks up.
    """
    name = f"{VOL_NAME}-{uuid.uuid4().hex[:8]}"
    logger.info("snapshot_repo: name=%s url=%s", name, tarball_url)

    vol = Volume.new(name=name)
    await vol.mount(mount_path="/workspace")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tarball = tmp.name
    await asyncio.to_thread(urllib.request.urlretrieve, tarball_url, tarball)
    n = await asyncio.to_thread(_extract_stripped, tarball, "/workspace")
    logger.info("snapshot_repo: extracted %d entries into /workspace", n)

    sealed = await vol.finalize(message=f"snapshot of {tarball_url}")
    return sealed, sealed.model_dump_json()


# ---------------------------------------------------------------------------
# vol.serve() PROTOTYPE — userspace volume serving, no FUSE
#
# Reuses the plugin's own internals (bundled juicefs binary + metadata-store
# materialization) to run `juicefs webdav` against the sealed index. This is
# everything ROVolume.mount() does *except* the kernel mount(2) — which is
# the only step that needs /dev/fuse + privileges. Candidate for promotion
# into flyteplugins-union as `ROVolume.serve()`.
# ---------------------------------------------------------------------------

WEBDAV_ADDR = os.environ.get("WEBDAV_ADDR", "127.0.0.1:9007")
WEBDAV_URL = f"http://{WEBDAV_ADDR}"


async def serve_volume_webdav(vol: ROVolume, addr: str = WEBDAV_ADDR, timeout: float = 60.0):
    """Serve ``vol`` read-only over WebDAV at ``addr`` — userspace only.

    Mirrors the front half of :meth:`Volume.mount` (index download +
    metadata materialization), then spawns ``juicefs webdav --read-only``
    instead of ``juicefs mount``. Returns the subprocess handle.
    """
    from flyteplugins.union.io._internal._juicefs._backend import _juicefs_binary

    store = vol._backend().store(vol._store_type())

    meta_dir = tempfile.mkdtemp(prefix="vol-serve-meta-")
    cache_dir = tempfile.mkdtemp(prefix="vol-serve-cache-")

    if vol.index is None:
        raise ValueError(f"volume {vol.name!r} has no published index — only sealed volumes can be served")

    # Download the published index and materialize it into the meta dir —
    # identical to the mount() path.
    with tempfile.NamedTemporaryFile(prefix="vol-serve-", suffix="-" + store.published_filename(), delete=False) as f:
        snapshot_path = f.name
    await vol.index.download(snapshot_path)
    await asyncio.to_thread(store.materialize, snapshot_path, meta_dir)
    meta_url = store.meta_url(meta_dir, await store.start(meta_dir))

    cmd = [
        _juicefs_binary(),
        "webdav",
        meta_url,
        addr,
        "--read-only",  # sealed volume: lookup/read only
        "--no-bgjob",  # no cleanup/backup jobs against an immutable index
        "--backup-meta",
        "0",
        "--cache-dir",
        cache_dir,
        "--log",
        os.path.join(tempfile.gettempdir(), "juicefs-webdav.log"),
    ]
    logger.info("serve_volume_webdav: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(*cmd)

    # Wait for the server to accept requests.
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if proc.returncode is not None:
            raise RuntimeError(f"juicefs webdav exited early with rc={proc.returncode}")
        try:
            req = urllib.request.Request(f"http://{addr}/", method="OPTIONS")
            await asyncio.to_thread(urllib.request.urlopen, req, None, 2)
            break
        except (urllib.error.URLError, OSError):
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError(f"juicefs webdav did not come up on {addr} within {timeout}s")
            await asyncio.sleep(0.5)
    logger.info("serve_volume_webdav: volume %s live at http://%s (read-only)", vol.name, addr)
    return proc


# --- Minimal stdlib WebDAV client (PROPFIND listings + GET) ----------------

_DAV = "{DAV:}"


def _dav_list(subpath: str) -> Tuple[bool, List[dict]]:
    """PROPFIND Depth:1 on ``subpath`` -> ``(is_dir, entries)``.

    PROPFIND on a *file* still succeeds (207 with the file's own props),
    so file-vs-directory is decided from the self entry's resourcetype,
    not from the status code. ``entries`` excludes the self entry and is
    empty when the target is a file.
    """
    url = f"{WEBDAV_URL}/{urllib.parse.quote(subpath)}"
    req = urllib.request.Request(url, method="PROPFIND", headers={"Depth": "1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        tree = ET.fromstring(r.read())

    self_path = "/" + subpath.strip("/")
    is_dir = True
    entries = []
    for resp in tree.findall(f"{_DAV}response"):
        href = urllib.parse.unquote(resp.findtext(f"{_DAV}href") or "")
        path = "/" + href.strip("/")
        prop = resp.find(f"{_DAV}propstat/{_DAV}prop")
        is_collection = prop is not None and prop.find(f"{_DAV}resourcetype/{_DAV}collection") is not None
        if path == self_path:  # the target itself
            is_dir = is_collection
            continue
        size = int(prop.findtext(f"{_DAV}getcontentlength") or 0) if prop is not None else 0
        entries.append({"name": Path(path).name, "is_dir": is_collection, "size": size})
    return is_dir, entries


def _dav_get(subpath: str) -> bytes:
    url = f"{WEBDAV_URL}/{urllib.parse.quote(subpath)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


# ---------------------------------------------------------------------------
# File browser app
# ---------------------------------------------------------------------------

app = FastAPI(title="Volume File Browser", description="Read-only browser over a sealed Flyte Volume — no FUSE")


def _render_dir(entries: List[dict], subpath: str) -> str:
    crumbs, acc = ['<a href="/browse/">root</a>'], ""
    for part in [p for p in subpath.split("/") if p]:
        acc = f"{acc}/{part}" if acc else part
        crumbs.append(f'<a href="/browse/{html.escape(acc, quote=True)}">{html.escape(part)}</a>')
    entries = sorted(entries, key=lambda e: (not e["is_dir"], e["name"].lower()))
    rows = []
    for e in entries:
        rel = f"{subpath.strip('/')}/{e['name']}".strip("/")
        label = html.escape(e["name"]) + ("/" if e["is_dir"] else "")
        size = "" if e["is_dir"] else f"{e['size']:,}"
        rows.append(
            f'<tr><td><a href="/browse/{html.escape(rel, quote=True)}">{label}</a></td><td align=right>{size}</td></tr>'
        )
    return (
        "<html><head><title>Volume File Browser</title></head>"
        "<body style='font-family:system-ui,sans-serif;max-width:60em;margin:2em auto'>"
        f"<h2>{' / '.join(crumbs)}</h2>"
        "<table cellpadding=4><tr><th align=left>name</th><th align=right>bytes</th></tr>"
        f"{''.join(rows)}</table>"
        f"<p><em>{len(entries)} entries — served read-only from a sealed Flyte Volume "
        "via userspace WebDAV (no FUSE, no privileges)</em></p>"
        "</body></html>"
    )


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/browse/")


@app.get("/health")
async def health() -> dict:
    serving = getattr(app.state, "volume", None) is not None
    return {"status": "ready" if serving else "starting", "webdav": WEBDAV_URL}


@app.get("/browse/{subpath:path}")
async def browse(subpath: str = ""):
    if ".." in subpath.split("/"):  # path-traversal guard
        raise HTTPException(status_code=403, detail="path escapes the volume root")
    try:
        is_dir, entries = await asyncio.to_thread(_dav_list, subpath)
        if not is_dir:
            content = await asyncio.to_thread(_dav_get, subpath)
            media_type = mimetypes.guess_type(subpath)[0] or "text/plain; charset=utf-8"
            return Response(content=content, media_type=media_type)
        return HTMLResponse(_render_dir(entries, subpath))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(status_code=404, detail=f"{subpath!r} not found")
        raise HTTPException(status_code=502, detail=f"webdav error: {e}")


env = FastAPIAppEnvironment(
    name="volume-file-browser",
    app=app,
    description="Scale-from-zero file browser over a Volume snapshot of the flyte-sdk repo (userspace, no FUSE)",
    image=image,
    # No enable_fuse_mount, no PodTemplate, no privileges: the userspace
    # WebDAV path is the whole point — this runs on any stock Knative.
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    requires_auth=False,
    # Scale-from-zero (the default, spelled out): no replicas when idle, one on
    # demand, scaled back down 5 minutes after the last request. Every cold
    # start re-runs on_startup — one index download + a subprocess spawn.
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
    parameters=[
        Parameter(
            # 'volume_spec' matches the argument name in @env.on_startup below.
            name="volume_spec",
            # Latest successful snapshot_repo run; getter=(1,) selects the JSON
            # spec (output o1) — app parameters carry strings, not Volumes.
            value=RunOutput(task_name="repo-snapshot.snapshot_repo", type="string", getter=(1,)),
            type="string",
            download=False,
        ),
    ],
    links=[flyte.app.Link(path="/browse/", title="Browse the volume", is_relative=True)],
)


@env.on_startup
async def startup(volume_spec: str) -> None:
    """Rebuild the sealed ROVolume from its JSON spec and serve it — userspace."""
    vol = ROVolume.model_validate_json(volume_spec)
    logger.info("startup: serving volume name=%s over webdav at %s", vol.name, WEBDAV_URL)
    app.state.webdav_proc = await serve_volume_webdav(vol)
    app.state.volume = vol
    logger.info("startup: volume %s is live", vol.name)


@env.on_shutdown
async def shutdown() -> None:
    proc = getattr(app.state, "webdav_proc", None)
    if proc is not None and proc.returncode is None:
        proc.terminate()
        await proc.wait()


# ---------------------------------------------------------------------------
# Main: snapshot the repo, then deploy the browser
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(snapshot_repo)
    print(f"Snapshot run: {run.url}")
    run.wait()

    deployed = flyte.deploy(env)
    print(f"App deployed: {deployed[0].table_repr()}")
