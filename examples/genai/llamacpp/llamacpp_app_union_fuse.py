"""
Serve a GGUF **Model artifact** with llama.cpp from a **Union Volume (JuiceFS)** -- the
`flyteplugins-union` device-plugin FUSE delivery mode, as opposed to the read-only
object-store PVC in `llamacpp_app_fuse.py`.

Three steps -- prefetch Artifact -> JuiceFS Volume -> serve:

  1. `flyte.prefetch.hf_model` publishes the weights as a versioned Model **artifact** (the
     source of truth + lineage), exactly like the other llama.cpp examples.
  2. `build_model_volume` mounts a fresh **RWVolume** (`Volume.new()` + the device-plugin FUSE
     via `PodTemplate.allow_fuse()`), copies the artifact's weights into it, and `commit()`s --
     returning an immutable **ROVolume locator**. One built volume fans out to many read-only
     serve replicas, backed by a JuiceFS local cache (fast repeat reads).
  3. The Flyte App runs a self-contained command that reconstructs `ROVolume.from_locator(locator)`,
     mounts it read-only, and execs `llama-server` via the plugin's `build_fserve_command` (the
     same argv the App/sidecar run, so serving stays in lockstep). See the note on step 3 below
     for why this is a command rather than an `@app.server` function.

Union Volume (JuiceFS) vs the object-store RO PVC (`llamacpp_app_fuse.py`):
  * Both deliver weights over FUSE without a per-start download. The Volume is a POSIX fs over
    object storage (immutable data chunks + a metadata index) with a writeback/local cache and
    multi-attach read-only; its `locator` rides the Flyte literal system, so lineage / caching /
    fork / clone are first-class.
  * The Volume needs the dataplane **union device-plugin** (the `fuseDevicePlugin` DaemonSet:
    `CAP_SYS_ADMIN` + the `smarter-devices/fuse` extended resource). `allow_fuse()` is the
    Knative-compatible *unprivileged* FUSE path (no privileged container, no `/dev/fuse`
    hostPath -- both rejected by Knative Serving).
  * Tradeoff: the device-plugin mount keeps a JuiceFS client subprocess alive for the pod's
    life, so -- unlike the RO-PVC fuse App -- this does **not** cleanly scale to zero. Keep at
    least one replica warm (`replicas=(1, ...)`).

Everything is env-configurable (`LLAMACPP_*`), defaulting to a small CPU model; set
`LLAMACPP_GPU` (e.g. `L4:1`) + a CUDA image for GPU serving.

Run:
    python examples/genai/llamacpp/llamacpp_app_union_fuse.py
"""

from __future__ import annotations

import asyncio
import os

import flyte
import flyte.app
from flyte.io import Dir

# Model + artifact identity (env-configurable; small CPU default for quick iteration).
MODEL_REPO = os.getenv("LLAMACPP_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
QUANT = os.getenv("LLAMACPP_QUANT", "q4_k_m")
ARTIFACT_NAME = os.getenv("LLAMACPP_ARTIFACT_NAME", "qwen2-5-0-5b-instruct-q4-k-m")
MODEL_ID = os.getenv("LLAMACPP_MODEL_ID", "qwen2.5-0.5b-instruct")
# Served Knative service is `<project>-<domain>-<APP_NAME>` (<= 63 chars). JuiceFS volume names
# allow only [a-z0-9-], 3-63 chars, so APP_NAME doubles as the volume name.
APP_NAME = os.getenv("LLAMACPP_APP_NAME", "qwen2-5-0-5b-instruct-vol")

SERVER_PORT = 8080
# Where the Volume is mounted inside the pod, and the JuiceFS local cache dir (on the serving
# node's ephemeral/NVMe disk, so repeat reads of the weights are fast).
MODEL_MOUNT = "/tmp/models"
CACHE_DIR = "/tmp/jfs-cache"

# GPU for serving (default CPU). Set e.g. "L4:1" for a GPU; EXTRA_ARGS then offloads layers.
GPU = os.getenv("LLAMACPP_GPU", "") or None
EXTRA_ARGS = os.getenv("LLAMACPP_EXTRA_ARGS", "--n-gpu-layers 999 --flash-attn on" if GPU else "--ctx-size 8192")
CPU = os.getenv("LLAMACPP_CPU", "2")
MEMORY = os.getenv("LLAMACPP_MEMORY", "8Gi")
DISK = os.getenv("LLAMACPP_DISK", "20Gi")

# Env var the serve() reads the committed volume's locator from (a plain string, injected at
# deploy; AppEnvironment parameters carry only file/dir/string, and a Volume is neither).
_LOCATOR_ENV = "LLAMACPP_MODEL_VOLUME_LOCATOR"


# ---- 2. Build a JuiceFS RWVolume from the prefetched artifact -----------------------------------

# Builder image: flyte + flyteplugins-union (JuiceFS client) + fuse3 (`fusermount3`), so the
# task can mount an RWVolume via the device-plugin path and stream the copied weights to object
# storage as immutable chunks. Deliberately NO flyteplugins-llamacpp: the builder task shares
# this module, so its container imports the whole file -- keep all llama.cpp (plugin) use out of
# module scope (the serve image is built in __main__) so the builder needn't carry the plugin.
builder_image = (
    flyte.Image.from_debian_base(name="llamacpp-volume-builder", install_flyte=True)
    .with_apt_packages("fuse3")
    .with_pip_packages("flyteplugins-union")
)
builder_env = flyte.TaskEnvironment(
    name="llamacpp-volume-builder",
    image=builder_image,
    # Unprivileged device-plugin FUSE (CAP_SYS_ADMIN + smarter-devices/fuse); needs the
    # dataplane fuseDevicePlugin DaemonSet. JuiceFS streams chunks to object storage as they are
    # written, so the builder needs only cache headroom, not the whole model on local disk.
    pod_template=flyte.PodTemplate().allow_fuse(),
    resources=flyte.Resources(cpu="2", memory="8Gi", disk="30Gi"),
)


@builder_env.task(cache="auto")
async def build_model_volume(model: Dir, volume_name: str) -> str:
    """Copy a prefetched Model artifact into a fresh Union Volume; return its ROVolume locator.

    `cache="auto"` keys on (model, volume_name): the same artifact + name is built once and
    reused. `model` is the prefetched artifact (bound as a task input, so the build Run records
    Run -> artifact lineage); its bytes are copied into the volume, then committed as immutable,
    multi-attach read-only chunks addressed by the returned `locator`.
    """
    from flyteplugins.union.io import Volume

    vol = Volume.new(name=volume_name)
    mount_path = await vol.mount()
    # Materialize the artifact's directory straight into the volume mount (JuiceFS uploads the
    # chunks as they land); no HF re-download -- the artifact is the source of truth.
    await model.download(str(mount_path))
    # commit() drains the writeback queue and returns the published, immutable ROVolume; the
    # locator lives on the returned value, not on the RWVolume handle.
    ro = await vol.commit()
    if not ro.locator:
        raise RuntimeError("committed volume has no locator")
    flyte.logger.info("Built model volume %r -> locator=%s", volume_name, ro.locator)
    return ro.locator


# ---- 3. Serve the model from a read-only Union Volume -------------------------------------------

# The App runs a self-contained **command** (not an `@app.server` function): mount the read-only
# Volume from its locator, then exec llama-server. A command is deliberate here -- a server
# *function* would need the app-serde resolver to re-import this module in the serving container
# to reload the function, which does not work when the example is run as a `python script.py`
# (`__main__`) program (the resolver can't turn `__main__` into an importable module -> the
# function is never loaded and fserve exits "No command provided"). A command carries everything
# in `args`, so it needs no resolver -- the same reason the App/sidecar examples use one. See the
# tracking issue for making the SDK fail-fast (or support) the server-function-from-__main__ case.
#
# The command imports flyteplugins.{union.io,llamacpp} at *runtime in the serve container* (both
# are in the serve image); keeping those imports inside the command string (not module scope)
# also means the builder task -- which shares this module -- never imports them.
_SERVE_SCRIPT = f"""\
import os, asyncio, subprocess
from flyteplugins.union.io import ROVolume
from flyteplugins.llamacpp import build_fserve_command
# Mount the RO Volume from its locator; keep `vol` referenced (blocking below) so the JuiceFS
# client subprocess backing the mount stays alive for llama-server's lifetime.
vol = asyncio.run(ROVolume.from_locator(os.environ[{_LOCATOR_ENV!r}]))
root = str(asyncio.run(vol.mount(mount_path={MODEL_MOUNT!r}, cache_dir={CACHE_DIR!r}, read_only=True)))
cmd = build_fserve_command(model_id={MODEL_ID!r}, port={SERVER_PORT}, model_dir=root, extra_args={EXTRA_ARGS.split()!r})
subprocess.run(["/bin/sh", "-c", " ".join(cmd)], check=True)
"""

app_env = flyte.app.AppEnvironment(
    name=APP_NAME,
    # `image="auto"` is a placeholder overwritten in __main__ (the llama.cpp serve image is built
    # there, not at module scope, so the builder task's container never imports flyteplugins.llamacpp).
    image="auto",
    port=SERVER_PORT,
    type="llama.cpp",
    command=["python", "-c", _SERVE_SCRIPT],
    # Unprivileged device-plugin FUSE for the read-only Volume mount (Knative-compatible).
    # The App-serde requires the primary container to be named "app" (allow_fuse creates it).
    pod_template=flyte.PodTemplate(primary_container_name="app").allow_fuse(),
    resources=flyte.Resources(cpu=CPU, memory=MEMORY, gpu=GPU, disk=DISK),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    # The JuiceFS client subprocess pins the pod, so keep >=1 replica (no clean scale-to-zero).
    scaling=flyte.app.Scaling(replicas=(1, 1)),
    requires_auth=True,
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Artifact

    flyte.init_from_config()

    # 1. Prefetch the weights as a versioned Model artifact (source of truth + lineage).
    run = flyte.prefetch.hf_model(
        repo=MODEL_REPO,
        artifact_name=ARTIFACT_NAME,
        allow_patterns=[f"*{QUANT}*"],
        hf_token_key=None,  # public repo: prefetch anonymously
        resources=flyte.Resources(
            cpu=os.getenv("LLAMACPP_PREFETCH_CPU", "2"),
            memory=os.getenv("LLAMACPP_PREFETCH_MEMORY", "4Gi"),
            disk=os.getenv("LLAMACPP_PREFETCH_DISK", "10Gi"),
        ),
    )
    print(f"Prefetching {MODEL_REPO} ({QUANT}) -> artifact {ARTIFACT_NAME!r}: {run.url}")
    run.wait()
    artifact: Artifact = asyncio.run(Artifact.get.aio(ARTIFACT_NAME, "latest"))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    # 2. Build the Union Volume from that artifact; its locator is the serve handle.
    build_run = flyte.run(build_model_volume, model=artifact, volume_name=APP_NAME)
    print(f"Building model volume: {build_run.url}")
    build_run.wait()
    locator = build_run.outputs()[0]  # single string output (ActionOutputs is a tuple)
    print(f"Model volume locator: {locator}")

    # 3. Build the serve image (llama.cpp + JuiceFS client + fuse3) -- imported here, not at
    #    module scope, so the builder task's container never imports flyteplugins.llamacpp.
    from flyteplugins.llamacpp import build_llama_cpp_image

    app_env.image = (
        build_llama_cpp_image(name="llamacpp-volume-serve", cuda=bool(GPU))
        .with_apt_packages("fuse3")
        .with_pip_packages("flyteplugins-union")
    )
    # 4. Deploy the App with the locator injected (serve() mounts the ROVolume from it).
    app_env.env_vars = {_LOCATOR_ENV: str(locator)}
    app = flyte.serve(app_env)
    print(f"Deployed llama.cpp app serving the {APP_NAME!r} Union Volume: {app.url}")
