"""
Speculative decoding with llama.cpp, served from a single Union Volume.

Speculative decoding pairs a large **target** model with a small **draft** model of the
same family (shared tokenizer/vocab): the draft proposes several tokens cheaply and the
target verifies them in one pass, so latency drops with no change to the output
distribution. llama.cpp exposes this via ``--model`` + ``--model-draft`` (here the
``llama-cpp-fserve`` shim's ``--model-dir`` / ``--draft-model-dir``).

This is the union-volume delivery of `llamacpp_app_union_volume.py`, extended to two
models. Both GGUFs live in **one committed Union Volume** under ``model/`` and ``draft/``
subdirs -- a single immutable ROVolume the App mounts read-only via the uvol broker
(zero-privilege inline CSI, Knative-1.23 compatible), so it still scales to zero cleanly.

Steps -- prefetch two Artifacts -> build one Union Volume -> serve:
  1. `flyte.prefetch.hf_model` publishes the target and draft weights as versioned Model
     artifacts (reused if already present).
  2. `build_spec_volume` copies both into one fresh Volume (``model/`` + ``draft/``) and
     commits it to an immutable ROVolume locator.
  3. The App runs `union-volume-exec` (mounts the ROVolume at ``/tmp/models``) wrapping the
     `build_fserve_command` argv with both ``--model-dir`` and ``--draft-model-dir``.

Defaults to a heavy pair sized for one L40S (48GB): Qwen2.5-32B target (q6_k) + 0.5B draft,
`LLAMACPP_GPU=L40S:1`. Override `LLAMACPP_GPU=""` for a (slow) CPU run, or point the
`LLAMACPP_*` vars at any same-family target/draft pair.

Run:
    python examples/genai/llamacpp/llamacpp_app_union_volume_speculative.py
"""

from __future__ import annotations

import asyncio
import os

from flyteplugins.union.io import allow_volumes

import flyte
import flyte.app
from flyte.io import Dir

# Target (large) and draft (small) must share a tokenizer/vocab -- same model family.
# Default target is Qwen2.5-32B (q6_k ~27GB weights + a large KV cache) sized to fill a single
# L40S (48GB VRAM); the 0.5B draft shares the Qwen2.5 vocab and costs <1GB. Speculative decoding
# shines here: the big target dominates cost, so cheap draft proposals cut latency the most.
TARGET_REPO = os.getenv("LLAMACPP_MODEL_REPO", "Qwen/Qwen2.5-32B-Instruct-GGUF")
TARGET_QUANT = os.getenv("LLAMACPP_QUANT", "q6_k")
TARGET_ARTIFACT = os.getenv("LLAMACPP_ARTIFACT_NAME", "qwen2-5-32b-instruct-q6-k")

DRAFT_REPO = os.getenv("LLAMACPP_DRAFT_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
DRAFT_QUANT = os.getenv("LLAMACPP_DRAFT_QUANT", "q4_k_m")
DRAFT_ARTIFACT = os.getenv("LLAMACPP_DRAFT_ARTIFACT_NAME", "qwen2-5-0-5b-instruct-q4-k-m")

MODEL_ID = os.getenv("LLAMACPP_MODEL_ID", "qwen2.5-32b-instruct")
APP_NAME = os.getenv("LLAMACPP_APP_NAME", "qwen2-5-32b-instruct-spec-vol")

SERVER_PORT = 8080
MODEL_MOUNT = "/tmp/models"  # union-volume-exec mounts the volume here
MODEL_SUBDIR = "model"  # target GGUF -> {MODEL_MOUNT}/model
DRAFT_SUBDIR = "draft"  # draft GGUF  -> {MODEL_MOUNT}/draft

# GPU-class by default: a 32B target needs a real accelerator. L40s:1 (managed EKS g6e) fits the
# q6_k weights + a 16k KV cache in one 48GB card. Override LLAMACPP_GPU="" to force CPU (slow).
# NB the accelerator token is case-sensitive to flyte.Resources -- it is "L40s:1" (lowercase s).
GPU = os.getenv("LLAMACPP_GPU", "L40s:1") or None
# Speculative decoding is enabled purely by passing a draft model (--model-draft, set by the shim
# from --draft-model-dir); llama.cpp uses sensible defaults for how many tokens it drafts per step.
# The draft-count tuning flags were renamed across llama.cpp versions (e.g. --draft-max ->
# --spec-draft-n-max), so they are NOT hardcoded here -- add them via LLAMACPP_EXTRA_ARGS if your
# llama.cpp build supports them.
EXTRA_ARGS = os.getenv(
    "LLAMACPP_EXTRA_ARGS",
    ("--n-gpu-layers 999 --flash-attn on --ctx-size 16384" if GPU else "--ctx-size 8192"),
)
CPU = os.getenv("LLAMACPP_CPU", "8")
MEMORY = os.getenv("LLAMACPP_MEMORY", "48Gi")
DISK = os.getenv("LLAMACPP_DISK", "80Gi")


def _volume_name(*identity: str) -> str:
    """JuiceFS-legal volume name derived from the (target, draft) artifact identities.

    A digest over both name@version pairs keys the volume, so changing either model yields
    a fresh volume automatically (``Volume.new`` is create-only -- this avoids the "Storage
    not empty" re-format collision).
    """
    import hashlib
    import re

    digest = hashlib.sha256("|".join(identity).encode()).hexdigest()[:12]
    prefix = re.sub(r"[^a-z0-9-]", "-", identity[0].lower()).strip("-")[: 63 - len(digest) - 6].strip("-")
    return f"{prefix or 'vol'}-spec-{digest}"


def _prebuilt_serve_image(cuda: bool) -> flyte.Image:
    """Serve image from the official prebuilt llama.cpp binaries (ggml-org), no source compile.

    ``LLAMACPP_PREBUILT=1`` opts in: ``from_base`` the official image (ships ``llama-server`` at
    ``/app``) and layer Python + the plugins on top. ``from_base`` images are unnamed and
    non-extendable by default, hence ``.clone(name=, extendable=)``.
    """
    base = "ghcr.io/ggml-org/llama.cpp:server-cuda" if cuda else "ghcr.io/ggml-org/llama.cpp:server"
    return (
        flyte.Image.from_base(base)
        .clone(name="llama-cpp-prebuilt", extendable=True)
        .with_apt_packages("python3", "python3-pip", "python3-venv")
        .with_pip_packages("flyteplugins-llamacpp", "flyteplugins-union>=0.11.0b0", pre=True)
        .with_env_vars({"PATH": "/app:/usr/local/bin:/usr/bin:/bin"})  # /app holds the prebuilt llama-server
    )


# The builder needs flyteplugins-union (the Volume client), not the llama.cpp plugin.
builder_image = flyte.Image.from_debian_base(name="llamacpp-spec-volume-builder", install_flyte=True).with_pip_packages(
    # Pinned to the beta that first ships `union-volume-exec`; relax once a stable 0.11.x is released.
    "flyteplugins-union>=0.11.0b0"
)
builder_env = flyte.TaskEnvironment(
    name="llamacpp-spec-volume-builder",
    image=builder_image,
    pod_template=allow_volumes(),  # zero-privilege brokered volume mount
    resources=flyte.Resources(cpu="2", memory="8Gi", disk="40Gi"),
)


@builder_env.task(cache="auto")
async def build_spec_volume(model: Dir, draft: Dir, volume_name: str) -> str:
    """Copy target + draft artifacts into one fresh Union Volume; return its ROVolume locator.

    ``cache="auto"`` keys on (model, draft, volume_name); both artifacts bind as task inputs so
    the Run records Run -> artifact lineage for each.
    """
    from flyteplugins.union.io import Volume

    vol = Volume.new(name=volume_name)
    mount_path = await vol.mount()
    await model.download(f"{mount_path}/{MODEL_SUBDIR}")
    await draft.download(f"{mount_path}/{DRAFT_SUBDIR}")
    ro = await vol.commit()
    if not ro.locator:
        raise RuntimeError("committed volume has no locator")
    flyte.logger.info("Built spec-decode volume %r -> locator=%s", volume_name, ro.locator)
    return ro.locator


# Command app; `image`/`args` are filled in __main__ once the serve image + volume locator exist.
app_env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image="auto",
    port=SERVER_PORT,
    type="llama.cpp",
    # Zero-privilege brokered mount of the read-only Volume; primary_container_name="app" required.
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

    def _ensure(repo: str, artifact_name: str, quant: str) -> Artifact:
        force = os.getenv("LLAMACPP_FORCE_PREFETCH", "").lower() in ("1", "true", "yes")
        if force or not _artifact_exists(artifact_name):
            run = flyte.prefetch.hf_model(
                repo=repo,
                artifact_name=artifact_name,
                allow_patterns=[f"*{quant}*"],
                hf_token_key=None,  # public repos
                resources=flyte.Resources(cpu="2", memory="4Gi", disk="10Gi"),
            )
            print(f"Prefetching {repo} ({quant}) -> {artifact_name!r}: {run.url}")
            run.wait()
        else:
            print(f"Reusing artifact {artifact_name!r} (LLAMACPP_FORCE_PREFETCH=1 to re-create)")
        return asyncio.run(Artifact.get.aio(artifact_name, "latest"))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    target = _ensure(TARGET_REPO, TARGET_ARTIFACT, TARGET_QUANT)
    draft = _ensure(DRAFT_REPO, DRAFT_ARTIFACT, DRAFT_QUANT)

    # One volume keyed on both models; changing either rebuilds it automatically.
    volume_name = _volume_name(f"{TARGET_ARTIFACT}@{target.version}", f"{DRAFT_ARTIFACT}@{draft.version}")
    build_run = flyte.run(build_spec_volume, model=target, draft=draft, volume_name=volume_name)
    print(f"Building spec-decode volume {volume_name!r}: {build_run.url}")
    build_run.wait()
    locator = build_run.outputs()[0]
    print(f"Model volume locator: {locator}")

    from flyteplugins.llamacpp import build_fserve_command, build_llama_cpp_image

    # Default: compile llama.cpp from source; `LLAMACPP_PREBUILT=1` uses the lighter prebuilt image.
    if os.getenv("LLAMACPP_PREBUILT", "").lower() in ("1", "true", "yes"):
        app_env.image = _prebuilt_serve_image(cuda=bool(GPU))
    else:
        app_env.image = build_llama_cpp_image(name="llamacpp-spec-serve", cuda=bool(GPU)).with_pip_packages(
            # Pinned to the beta that first ships `union-volume-exec`; relax once a stable 0.11.x is released.
            "flyteplugins-union>=0.11.0b0"
        )

    serve_cmd = build_fserve_command(
        model_id=MODEL_ID,
        port=SERVER_PORT,
        model_dir=f"{MODEL_MOUNT}/{MODEL_SUBDIR}",
        draft_model_dir=f"{MODEL_MOUNT}/{DRAFT_SUBDIR}",
        extra_args=EXTRA_ARGS.split(),
    )
    # union-volume-exec mounts the ROVolume at MODEL_MOUNT (holding model/ + draft/), then serves.
    app_env.args = ["union-volume-exec", "--locator", str(locator), "--mount-at", MODEL_MOUNT, "--", *serve_cmd]
    app = flyte.serve(app_env)
    print(f"Deployed speculative-decoding llama.cpp app {APP_NAME!r} (target+draft in one Union Volume): {app.url}")
