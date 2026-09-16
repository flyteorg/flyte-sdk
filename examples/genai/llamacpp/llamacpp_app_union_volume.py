"""
Serve a GGUF Model artifact with llama.cpp from a Union Volume, mounted via the uvol mount
broker (`allow_volumes()`) -- the alternative to the read-only object-store PVC in
`llamacpp_app_fuse.py`.

Steps -- prefetch Artifact -> build Union Volume -> serve:
  1. `flyte.prefetch.hf_model` publishes the weights as a versioned Model artifact.
  2. `build_model_volume` copies that artifact into a fresh Volume and commits it to an
     immutable ROVolume locator (one build fans out to many read-only serve replicas).
  3. The App runs `union-volume-exec` (mounts the ROVolume read-only) wrapping the
     `build_fserve_command` argv -- a command app, like the other llama.cpp examples.

The broker mounts volumes zero-privilege via inline ephemeral CSI (no CAP_SYS_ADMIN, no
/dev/fuse, no hostPath), Knative-serving compatible with the 1.23 gateway. The mount releases
when the pod scales down, so this scales to zero cleanly like the RO-PVC example.

Everything is env-configurable (`LLAMACPP_*`), defaulting to a small CPU model; set
`LLAMACPP_GPU` (e.g. `L4:1`) + a CUDA image for GPU serving.

Run:
    python examples/genai/llamacpp/llamacpp_app_union_volume.py
"""

from __future__ import annotations

import asyncio
import os

from flyteplugins.union.io import allow_volumes

import flyte
import flyte.app
from flyte.io import Dir

MODEL_REPO = os.getenv("LLAMACPP_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
QUANT = os.getenv("LLAMACPP_QUANT", "q4_k_m")
ARTIFACT_NAME = os.getenv("LLAMACPP_ARTIFACT_NAME", "qwen2-5-0-5b-instruct-q4-k-m")
MODEL_ID = os.getenv("LLAMACPP_MODEL_ID", "qwen2.5-0.5b-instruct")
APP_NAME = os.getenv("LLAMACPP_APP_NAME", "qwen2-5-0-5b-instruct-vol")


def _volume_name(artifact_name: str, artifact_version: str) -> str:
    """Volume name derived from the artifact identity -- one volume per (name, version).

    A digest of the full name@version keeps it unique and bounded even when ARTIFACT_NAME is
    long or unusual; the sanitized name is a best-effort readable prefix.
    """
    import hashlib
    import re

    digest = hashlib.sha256(f"{artifact_name}@{artifact_version}".encode()).hexdigest()[:12]
    prefix = re.sub(r"[^a-z0-9-]", "-", artifact_name.lower()).strip("-")[: 63 - len(digest) - 1].strip("-")
    return f"{prefix or 'vol'}-{digest}"


SERVER_PORT = 8080
MODEL_MOUNT = "/tmp/models"  # where union-volume-exec mounts the volume; --model-dir points here

GPU = os.getenv("LLAMACPP_GPU", "") or None
EXTRA_ARGS = os.getenv("LLAMACPP_EXTRA_ARGS", "--n-gpu-layers 999 --flash-attn on" if GPU else "--ctx-size 8192")
CPU = os.getenv("LLAMACPP_CPU", "2")
MEMORY = os.getenv("LLAMACPP_MEMORY", "8Gi")
DISK = os.getenv("LLAMACPP_DISK", "20Gi")


def _prebuilt_serve_image(cuda: bool) -> flyte.Image:
    """Serve image from the official prebuilt llama.cpp binaries (ggml-org) instead of compiling.

    `build_llama_cpp_image` compiles llama.cpp from source -- correct, but a heavy cmake/CUDA build.
    `LLAMACPP_PREBUILT=1` opts into this lighter alternative: `from_base` the official image (which
    ships `llama-server` at `/app`) and only layer Python + the plugins on top (seconds, no compile).
    `from_base` images are unnamed and non-extendable by default, hence `.clone(name=, extendable=)`.
    """
    base = "ghcr.io/ggml-org/llama.cpp:server-cuda" if cuda else "ghcr.io/ggml-org/llama.cpp:server"
    return (
        flyte.Image.from_base(base)
        .clone(name="llama-cpp-prebuilt", extendable=True)
        .with_apt_packages("python3", "python3-pip", "python3-venv")
        .with_pip_packages("flyteplugins-llamacpp", "flyteplugins-union>=0.11.0b0", pre=True)
        .with_env_vars({"PATH": "/app:/usr/local/bin:/usr/bin:/bin"})  # /app holds the prebuilt llama-server
    )


# The builder needs flyteplugins-union (the Volume client), not the llama.cpp plugin -- keep
# plugin imports out of module scope (serve image is built in __main__) so this task stays lean.
builder_image = flyte.Image.from_debian_base(name="llamacpp-volume-builder", install_flyte=True).with_pip_packages(
    # Pinned to the beta that first ships `union-volume-exec`; relax once a stable 0.11.x is released.
    "flyteplugins-union>=0.11.0b0"
)
builder_env = flyte.TaskEnvironment(
    name="llamacpp-volume-builder",
    image=builder_image,
    pod_template=allow_volumes(),  # zero-privilege brokered volume mount
    resources=flyte.Resources(cpu="2", memory="8Gi", disk="30Gi"),
)


@builder_env.task(cache="auto")
async def build_model_volume(model: Dir, volume_name: str) -> str:
    """Copy the prefetched Model artifact into a fresh Union Volume; return its ROVolume locator.

    `cache="auto"` keys on (model, volume_name), and `model` is bound as a task input so the Run
    records Run -> artifact lineage.
    """
    from flyteplugins.union.io import Volume

    vol = Volume.new(name=volume_name)
    mount_path = await vol.mount()
    await model.download(str(mount_path))
    ro = await vol.commit()
    if not ro.locator:
        raise RuntimeError("committed volume has no locator")
    flyte.logger.info("Built model volume %r -> locator=%s", volume_name, ro.locator)
    return ro.locator


# Command app (like vLLM/SGLang), not an `@app.server` function; `image`/`args` are filled in
# __main__ once the serve image is built and the volume locator is known.
app_env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image="auto",
    port=SERVER_PORT,
    type="llama.cpp",
    # Zero-privilege brokered mount of the read-only Volume; primary_container_name="app" is
    # required by the App-serde.
    pod_template=allow_volumes(flyte.PodTemplate(primary_container_name="app")),
    resources=flyte.Resources(cpu=CPU, memory=MEMORY, gpu=GPU, disk=DISK),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),  # scale to zero; broker releases with pod
    requires_auth=True,
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Artifact

    flyte.init_from_config()

    def _artifact_exists(name: str) -> bool:
        try:
            Artifact.get(name)
            return True
        except Exception:
            return False

    # Reuse the artifact if present; prefetch only when missing or LLAMACPP_FORCE_PREFETCH is set.
    force = os.getenv("LLAMACPP_FORCE_PREFETCH", "").lower() in ("1", "true", "yes")
    if force or not _artifact_exists(ARTIFACT_NAME):
        run = flyte.prefetch.hf_model(
            repo=MODEL_REPO,
            artifact_name=ARTIFACT_NAME,
            allow_patterns=[f"*{QUANT}*"],
            hf_token_key=None,  # public repo
            resources=flyte.Resources(
                cpu=os.getenv("LLAMACPP_PREFETCH_CPU", "2"),
                memory=os.getenv("LLAMACPP_PREFETCH_MEMORY", "4Gi"),
                disk=os.getenv("LLAMACPP_PREFETCH_DISK", "10Gi"),
            ),
        )
        print(f"Prefetching {MODEL_REPO} ({QUANT}) -> artifact {ARTIFACT_NAME!r}: {run.url}")
        run.wait()
    else:
        print(f"Reusing artifact {ARTIFACT_NAME!r} (LLAMACPP_FORCE_PREFETCH=1 to re-create)")
    artifact: Artifact = asyncio.run(Artifact.get.aio(ARTIFACT_NAME, "latest"))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    # Build the Volume from the artifact (name derived from its identity); its locator is the serve handle.
    volume_name = _volume_name(ARTIFACT_NAME, artifact.version)
    build_run = flyte.run(build_model_volume, model=artifact, volume_name=volume_name)
    print(f"Building model volume {volume_name!r} (artifact {ARTIFACT_NAME}@{artifact.version}): {build_run.url}")
    build_run.wait()
    locator = build_run.outputs()[0]
    print(f"Model volume locator: {locator}")

    # Serve image built here (not module scope) so the builder task never imports the llama.cpp plugin.
    from flyteplugins.llamacpp import build_fserve_command, build_llama_cpp_image

    # Default: compile llama.cpp from source; `LLAMACPP_PREBUILT=1` uses the lighter prebuilt image.
    if os.getenv("LLAMACPP_PREBUILT", "").lower() in ("1", "true", "yes"):
        app_env.image = _prebuilt_serve_image(cuda=bool(GPU))
    else:
        app_env.image = build_llama_cpp_image(name="llamacpp-volume-serve", cuda=bool(GPU)).with_pip_packages(
            # Pinned to the beta that first ships `union-volume-exec`; relax once a stable 0.11.x is released.
            "flyteplugins-union>=0.11.0b0"
        )
    serve_cmd = build_fserve_command(
        model_id=MODEL_ID,
        port=SERVER_PORT,
        model_dir=MODEL_MOUNT,
        extra_args=EXTRA_ARGS.split(),
    )
    # union-volume-exec mounts the ROVolume at MODEL_MOUNT, then runs the llama.cpp serve command.
    app_env.args = ["union-volume-exec", "--locator", str(locator), "--mount-at", MODEL_MOUNT, "--", *serve_cmd]
    app = flyte.serve(app_env)
    print(f"Deployed llama.cpp app serving the {APP_NAME!r} Union Volume: {app.url}")
