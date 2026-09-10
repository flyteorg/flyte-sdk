"""
Serve a GGUF **Model artifact** with llama.cpp, streamed in place over object-store FUSE
instead of downloaded.

This is the object-store-FUSE delivery mode; see `llamacpp_app.py` for the plain download
mode. Both serve the same prefetched model; they differ only in how the weights reach the
server:

  * `llamacpp_app.py`  -- `model_delivery="download"` (default): the bound artifact is copied
    into the pod's local disk before llama-server starts. Simple, but the whole GGUF lands on
    the node's ephemeral disk and the copy is on the cold-start path.
  * this example      -- `model_delivery="fuse"`: the weights are read **in place** from a
    read-only, object-store-backed PVC via a CSI driver. First touch is lazy (~20-25 s for an
    ~18 GB GGUF), nothing is copied to local disk, and the mount releases cleanly when the app
    scales to zero -- so scale-from-zero is bounded by GPU node cold-start, not by re-downloading
    the model.

The whole interface is one line: `model_path=ArtifactValue(name=..., type="directory")`. The
artifact is resolved to its object-store URI at deploy, streamed in place from the bucket-root
mount, and the App→artifact lineage edge is recorded automatically. There is no bucket prefix
to configure and no subpath to derive -- the artifact's own bucket *is* the mount.

Prerequisite -- the read-only model PVC
---------------------------------------
`model_delivery="fuse"` expects a valid, pre-provisioned read-only PVC that exposes the dataplane
**data bucket root** -- the same bucket the Model artifact materializes into -- so an artifact at
`<scheme>://<data-bucket>/<key>` is read in place at `<mount>/<key>`. You reference that claim by
name via `model_pvc` (required), so it must already exist in the app's namespace; on a managed
dataplane it is provisioned for you as `flyte-metadata-ro`. **README.md** carries example manifests
for both CSI backends (gcsfuse on GKE, Mountpoint-S3 on EKS) and the one field that differs between
them. gcsfuse also needs the `gke-gcsfuse/volumes: "true"` pod annotation (set via
`fuse_pod_annotations` below); Mountpoint-S3 needs none.

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

from flyteplugins.llamacpp import LlamaCppAppEnvironment

import flyte
import flyte.app

# Model + artifact identity are read from environment variables (with these defaults) so the
# example serves a different model without editing -- e.g. for qwen38-27b set
# LLAMACPP_MODEL_REPO / LLAMACPP_QUANT / LLAMACPP_ARTIFACT_NAME / LLAMACPP_MODEL_ID (and, if you
# want a distinct app, LLAMACPP_APP_NAME).
MODEL_REPO = os.getenv("LLAMACPP_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
QUANT = os.getenv("LLAMACPP_QUANT", "q4_k_m")
ARTIFACT_NAME = os.getenv("LLAMACPP_ARTIFACT_NAME", "qwen2-5-0-5b-instruct-q4-k-m")
MODEL_ID = os.getenv("LLAMACPP_MODEL_ID", "qwen2.5-0.5b-instruct")
# The served Knative service is named `<project>-<domain>-<APP_NAME>` and must be <= 63 chars, so
# keep this short (long project/domain names eat into the budget).
APP_NAME = os.getenv("LLAMACPP_APP_NAME", "qwen2-5-0-5b-instruct")

# Name of the pre-provisioned, read-only PVC exposing the data bucket root (created by the
# dataplane helm release, e.g. `flyte-metadata-ro`). Required: it must name a claim that already
# exists in the app's namespace -- see README.md for the gcsfuse (GKE) / Mountpoint-S3 (EKS) manifests.
MODEL_PVC = os.getenv("LLAMACPP_MODEL_PVC", "flyte-metadata-ro")

# GPU accelerator to request. The type is cloud-specific -- L4 on GKE, L40s on the AWS g6e pool,
# A10 on Azure -- so it reads from LLAMACPP_GPU to keep the example portable across dataplanes
# without editing. Set LLAMACPP_GPU="" for CPU-only (also pass a CPU image via
# build_llama_cpp_image(cuda=False) -- see README Variations). Use e.g. "L4:2" to request two
# GPUs; llama.cpp then splits layers across them (pass the split flags via LLAMACPP_EXTRA_ARGS).
GPU = os.getenv("LLAMACPP_GPU", "L4:1") or None

# Pod sizing + llama-server tuning, env-driven so the example scales from the 0.5B default to a
# multi-shard 27B without editing. IMPORTANT for GPU serving: llama-server defaults to CPU
# (`-ngl 0`); to actually use the GPU(s) pass `--n-gpu-layers 999` here. For a 2-GPU model add
# `--split-mode layer --tensor-split 1,1`. Example for Q8_0 27B on 2x L4:
#   LLAMACPP_EXTRA_ARGS="--ctx-size 16384 --n-gpu-layers 999 --split-mode layer --tensor-split 1,1 --flash-attn"
CPU = os.getenv("LLAMACPP_CPU", "2")
MEMORY = os.getenv("LLAMACPP_MEMORY", "8Gi")
DISK = os.getenv("LLAMACPP_DISK", "10Gi")
EXTRA_ARGS = os.getenv("LLAMACPP_EXTRA_ARGS", "--ctx-size 8192")

fuse_app = LlamaCppAppEnvironment(
    name=APP_NAME,
    model_id=MODEL_ID,
    # Lazy object-store-FUSE mount instead of a download: the weights are read in place from the
    # RO PVC, and the app scales to zero with the mount releasing cleanly.
    model_delivery="fuse",
    model_pvc=MODEL_PVC,
    # The Model artifact, served directly: resolved to its object-store URI at deploy, streamed
    # from the bucket-root mount, lineage recorded -- no bucket prefix, no subpath.
    model_path=flyte.app.ArtifactValue(name=ARTIFACT_NAME, type="directory"),
    # Required on gcsfuse (GKE) -- the sidecar injector only mounts the volume when the pod
    # carries this annotation. Harmless on Mountpoint-S3 (EKS), which ignores it, so it can be
    # left in for portability or dropped for cleanliness.
    fuse_pod_annotations={"gke-gcsfuse/volumes": "true"},
    resources=flyte.Resources(cpu=CPU, memory=MEMORY, gpu=GPU, disk=DISK),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,  # scale to zero after 5 minutes idle; the FUSE mount releases clean
    ),
    requires_auth=True,
    extra_args=EXTRA_ARGS,
)


if __name__ == "__main__":
    import flyte.prefetch

    flyte.init_from_config()

    # 1. A Flyte run creates the Model artifact. `hf_model` prefetches one quant (allow_patterns
    #    keeps it to the Q4_K_M file) and publishes it as a versioned artifact in the data bucket
    #    -- the same bucket the RO PVC mounts.
    run = flyte.prefetch.hf_model(
        repo=MODEL_REPO,
        artifact_name=ARTIFACT_NAME,
        allow_patterns=[f"*{QUANT}*"],
        hf_token_key=None,  # public repo: prefetch anonymously
        # Prefetch is CPU-only but disk/mem must hold the selected quant -- scale via env for a
        # large model (e.g. a ~27 GB Q8_0 needs LLAMACPP_PREFETCH_DISK=60Gi).
        resources=flyte.Resources(
            cpu=os.getenv("LLAMACPP_PREFETCH_CPU", "2"),
            memory=os.getenv("LLAMACPP_PREFETCH_MEMORY", "4Gi"),
            disk=os.getenv("LLAMACPP_PREFETCH_DISK", "10Gi"),
        ),
    )
    print(f"Prefetching {MODEL_REPO} ({QUANT}) -> artifact {ARTIFACT_NAME!r}: {run.url}")
    run.wait()

    # 2. Serve it via a lazy FUSE mount (no download). The app binds the artifact by name; at
    #    deploy it resolves to the URI this run just produced, and the shim streams it in place.
    app = flyte.serve(fuse_app)
    print(f"Deployed llama.cpp app streaming the {ARTIFACT_NAME!r} artifact over FUSE: {app.url}")
