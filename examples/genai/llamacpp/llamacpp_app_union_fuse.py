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
  3. The Flyte App composes two plugin shims (no cross-plugin dependency): `union-volume-exec`
     (from `flyteplugins-union`, which owns Volumes) mounts the `ROVolume` read-only at a local
     dir and then runs the wrapped command; that wrapped command is the plugin's
     `build_fserve_command` (from `flyteplugins-llamacpp`) pointed at the *same* dir -- the exact
     argv the download / object-store-fuse examples build, so serving stays in lockstep. Like
     vLLM/SGLang, this is a command app, not an `@app.server` function: llama-server is an external
     binary, so there is no Python server object for the app-serde resolver to re-import -- the
     argv carries everything.

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
# Served Knative service is `<project>-<domain>-<APP_NAME>` (<= 63 chars).
APP_NAME = os.getenv("LLAMACPP_APP_NAME", "qwen2-5-0-5b-instruct-vol")
# The Union Volume name (JuiceFS: [a-z0-9-], 3-63 chars). `Volume.new()` is CREATE-ONLY -- it
# `juicefs format`s a fresh storage prefix and FAILS ("Storage ... is not empty") if that prefix
# already holds data. The version suffix makes rebuilds explicit: with cache="auto", re-running
# the same version is a cache-hit (no rebuild, no collision); bump LLAMACPP_VOLUME_VERSION to
# build a fresh volume (e.g. the upstream weights changed).
VOLUME_VERSION = os.getenv("LLAMACPP_VOLUME_VERSION", "v1")
VOLUME_NAME = f"{APP_NAME}-{VOLUME_VERSION}"

SERVER_PORT = 8080
# Local dir the Union Volume is mounted at inside the serve pod (union-volume-exec mounts here;
# build_fserve_command points --model-dir at the same path).
MODEL_MOUNT = "/tmp/models"

# GPU for serving (default CPU). Set e.g. "L4:1" for a GPU; EXTRA_ARGS then offloads layers.
GPU = os.getenv("LLAMACPP_GPU", "") or None
EXTRA_ARGS = os.getenv("LLAMACPP_EXTRA_ARGS", "--n-gpu-layers 999 --flash-attn on" if GPU else "--ctx-size 8192")
CPU = os.getenv("LLAMACPP_CPU", "2")
MEMORY = os.getenv("LLAMACPP_MEMORY", "8Gi")
DISK = os.getenv("LLAMACPP_DISK", "20Gi")


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

# A command app (`args`), like vLLM/SGLang and the other llama.cpp examples -- not an
# `@app.server` function. `args` is filled in __main__ once the built volume's locator is known:
# `union-volume-exec` mounts the ROVolume then runs `build_fserve_command` pointed at the mount.
app_env = flyte.app.AppEnvironment(
    name=APP_NAME,
    # `image="auto"` is a placeholder overwritten in __main__ (the llama.cpp serve image is built
    # there, not at module scope, so the builder task's container never imports flyteplugins.llamacpp).
    image="auto",
    port=SERVER_PORT,
    type="llama.cpp",
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
    build_run = flyte.run(build_model_volume, model=artifact, volume_name=VOLUME_NAME)
    print(f"Building model volume: {build_run.url}")
    build_run.wait()
    locator = build_run.outputs()[0]  # single string output (ActionOutputs is a tuple)
    print(f"Model volume locator: {locator}")

    # 3. Build the serve image (llama.cpp + JuiceFS client + fuse3) -- imported here, not at
    #    module scope, so the builder task's container never imports flyteplugins.llamacpp.
    from flyteplugins.llamacpp import build_fserve_command, build_llama_cpp_image

    app_env.image = (
        build_llama_cpp_image(name="llamacpp-volume-serve", cuda=bool(GPU))
        .with_apt_packages("fuse3")
        .with_pip_packages("flyteplugins-union")
    )
    # 4. Compose the two shims: union-volume-exec mounts the volume at MODEL_MOUNT, then runs
    #    llama-cpp-fserve (build_fserve_command) pointed at that same dir.
    serve_cmd = build_fserve_command(
        model_id=MODEL_ID,
        port=SERVER_PORT,
        model_dir=MODEL_MOUNT,
        extra_args=EXTRA_ARGS.split(),
    )
    app_env.args = ["union-volume-exec", "--locator", str(locator), "--mount-at", MODEL_MOUNT, "--", *serve_cmd]
    app = flyte.serve(app_env)
    print(f"Deployed llama.cpp app serving the {APP_NAME!r} Union Volume: {app.url}")
