"""
Serve a GGUF **Model artifact** with llama.cpp, streamed in place over object-store FUSE
instead of downloaded.

This is the object-store-FUSE delivery mode; see `llamacpp_app.py` for the plain download
mode. Both serve the same prefetched model; they differ only in how the weights reach the
server:

  * `llamacpp_app.py`  -- `model_path` (default download): the bound artifact is copied
    into the pod's local disk before llama-server starts. Simple, but the whole GGUF lands on
    the node's ephemeral disk and the copy is on the cold-start path.
  * this example      -- `mount=ObjectStoreMount(...)`: the weights are read **in place** from a
    read-only, object-store-backed PVC via a CSI driver. First touch is lazy (~20-25 s for an
    ~18 GB GGUF), nothing is copied to local disk, and the mount releases cleanly when the app
    scales to zero -- so scale-from-zero is bounded by GPU node cold-start, not by re-downloading
    the model.

The whole interface is one line: `mount=ObjectStoreMount(pvc=..., model_path=ArtifactValue(...))`.
The artifact is resolved to its object-store URI at deploy, streamed in place from the bucket-root
mount, and the App→artifact lineage edge is recorded automatically. There is no bucket prefix
to configure and no subpath to derive -- the artifact's own bucket *is* the mount.

Prerequisite -- the read-only model PVC
---------------------------------------
`mount` expects a valid, pre-provisioned read-only PVC that exposes the dataplane
**data bucket root** -- the same bucket the Model artifact materializes into -- so an artifact at
`<scheme>://<data-bucket>/<key>` is read in place at `<mount>/<key>`. You reference that claim by
name via `ObjectStoreMount.pvc`, so it must already exist in the app's namespace; on a managed
dataplane it is provisioned for you as `flyte-metadata-ro`. **README.md** carries example manifests
for both CSI backends (gcsfuse on GKE, Mountpoint-S3 on EKS) and the one field that differs between
them. gcsfuse also needs the `gke-gcsfuse/volumes: "true"` pod annotation (set via
`pod_annotations` below); Mountpoint-S3 needs none.

Run -> Model artifact -> FUSE-streamed serve
-----------------------------------------------------------

```
python examples/genai/llamacpp/llamacpp_app_fuse.py
```

`__main__` shows the end-to-end loop: (1) a Flyte run (`hf_model`) creates a versioned Model
**artifact** in the data bucket; (2) the app binds that artifact by name and streams it in place
-- no download, no subpath wiring -- so the served weights are exactly the artifact this run
produced.

Or deploy the module-scope app directly (it resolves the latest artifact of that name):

```
flyte deploy examples/genai/llamacpp/llamacpp_app_fuse.py fuse_app
```

Usage is identical to `llamacpp_app.py` (OpenAI-compatible client against `<endpoint>/v1`).
"""

import os

from flyteplugins.llamacpp import LlamaCppAppEnvironment, ObjectStoreMount

import flyte
import flyte.app

# Defaults to Qwen3.8-27B Q8_0 (~27 GB, GPU-class); every field is LLAMACPP_*-overridable (point
# it at e.g. Qwen2.5-0.5B q4_k_m for quick iteration). APP_NAME is the Knative name + the model id.
MODEL_REPO = os.getenv("LLAMACPP_MODEL_REPO", "unsloth/Qwen3.8-27B-GGUF")
QUANT = os.getenv("LLAMACPP_QUANT", "Q8_0")
ARTIFACT_NAME = os.getenv("LLAMACPP_ARTIFACT_NAME", "qwen38-27b-q8-0")
APP_NAME = os.getenv("LLAMACPP_APP_NAME", "qwen38-27b")

# Pre-provisioned RO PVC exposing the data-bucket root (managed: `flyte-metadata-ro`). See README
# for the gcsfuse (GKE) / Mountpoint-S3 (EKS) manifests.
MODEL_PVC = os.getenv("LLAMACPP_MODEL_PVC", "flyte-metadata-ro")

# GPU is cloud-specific for ~48 GiB VRAM: 2x L4 (GKE), L40s:1 (AWS g6e), A10:2 (Azure). llama.cpp
# auto-splits layers across visible GPUs. Set "" for CPU-only (+ a CPU image, see README).
GPU = os.getenv("LLAMACPP_GPU", "L4:2") or None

# Pod sizing + llama-server tuning. --n-gpu-layers 999 offloads all layers to GPU (default is CPU);
# drop it (and shrink) for the small CPU model.
CPU = os.getenv("LLAMACPP_CPU", "8")
MEMORY = os.getenv("LLAMACPP_MEMORY", "48Gi")
# fuse mode: gcsfuse's file-cache is unbounded by default, so it grows to ~the model size on the
# pod's ephemeral storage as weights page in -- `disk` (the ephemeral limit) must hold it or the
# kubelet evicts the pod mid-load. Size >= the served quant (or bound the cache on the PVC).
DISK = os.getenv("LLAMACPP_DISK", "40Gi")
# `--flash-attn on` (recent llama.cpp requires the on|off|auto value, not a bare flag).
EXTRA_ARGS = os.getenv("LLAMACPP_EXTRA_ARGS", "--ctx-size 16384 --n-gpu-layers 999 --flash-attn on")

fuse_app = LlamaCppAppEnvironment(
    name=APP_NAME,
    # Lazy object-store-FUSE mount (read in place, no download; releases on scale-to-zero). The Model
    # artifact is served directly: resolved to its object-store URI at deploy, lineage recorded.
    mount=ObjectStoreMount(pvc=MODEL_PVC, model_path=flyte.app.ArtifactValue(name=ARTIFACT_NAME, type="directory")),
    # gcsfuse (GKE) mounts only with this annotation; Mountpoint-S3 (EKS) ignores it.
    pod_annotations={"gke-gcsfuse/volumes": "true"},
    # gpu is a free-form env str; Resources.gpu is a strict Literal, hence the type-ignore.
    resources=flyte.Resources(cpu=CPU, memory=MEMORY, gpu=GPU, disk=DISK),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,  # scale to zero after 5 minutes idle; the FUSE mount releases clean
    ),
    requires_auth=True,
    extra_args=EXTRA_ARGS,
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

    # Reuse the artifact; prefetch only when missing or LLAMACPP_FORCE_PREFETCH is set.
    force = os.getenv("LLAMACPP_FORCE_PREFETCH", "").lower() in ("1", "true", "yes")
    if force or not _artifact_exists(ARTIFACT_NAME):
        run = flyte.prefetch.hf_model(
            repo=MODEL_REPO,
            artifact_name=ARTIFACT_NAME,
            allow_patterns=[f"*{QUANT}*"],
            hf_token_key=None,  # public repo: prefetch anonymously
            # Disk must hold the quant (sized for the 27 GB default; shrink via LLAMACPP_PREFETCH_DISK).
            resources=flyte.Resources(
                cpu=os.getenv("LLAMACPP_PREFETCH_CPU", "4"),
                memory=os.getenv("LLAMACPP_PREFETCH_MEMORY", "16Gi"),
                disk=os.getenv("LLAMACPP_PREFETCH_DISK", "60Gi"),
            ),
        )
        print(f"Prefetching {MODEL_REPO} ({QUANT}) -> artifact {ARTIFACT_NAME!r}: {run.url}")
        run.wait()
    else:
        print(f"Reusing artifact {ARTIFACT_NAME!r} (LLAMACPP_FORCE_PREFETCH=1 to re-create)")

    # Serve: the app binds the artifact by name; deploy resolves + streams it in place over FUSE.
    app = flyte.serve(fuse_app)
    print(f"Deployed llama.cpp app streaming the {ARTIFACT_NAME!r} artifact over FUSE: {app.url}")
