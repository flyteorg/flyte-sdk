"""
Serve a GGUF model with llama.cpp on **GKE**, delivered by a lazy **gcsfuse** mount instead
of a download.

This is the GCP variant of the object-store-FUSE delivery mode; see `llamacpp_app_mountpoint_s3.py` for the
AWS (Mountpoint-S3) counterpart, and `llamacpp_app.py` for the plain download mode. All three
serve the *same* prefetched model; they differ only in how the weights reach the server:

  * `llamacpp_app.py`  -- `model_delivery="download"` (default): the bound artifact is copied
    into the pod's local disk before llama-server starts. Simple, but the whole GGUF lands on
    the node's ephemeral disk and the copy is on the cold-start path.
  * llamacpp_app_gcsfuse.py -- `model_delivery="fuse"` on GKE: the weights are read **in place**
    from a read-only, GCS-backed PVC via the gcsfuse CSI driver. First touch is lazy (~20-25 s
    for an ~18 GB GGUF), nothing is copied to local disk, and the mount releases cleanly when
    the app scales to zero -- so scale-from-zero is bounded by GPU node cold-start, not by
    re-downloading the model.

Why a PVC and not an inline CSI volume: a `PersistentVolumeClaim` is on Knative's podspec
allow-list and releases cleanly on scale-down; an inline CSI volume is not, and a node-level
device-plugin (JuiceFS / Union Volume) pins the node and blocks scale-to-zero. Object-store
FUSE via a static PVC is the only blob-backed delivery that is simultaneously lazy and
scale-to-zero-clean.

The one GKE-specific bit: the gcsfuse CSI runs its fuse process in a sidecar that the GKE
injector only adds when the pod carries `gke-gcsfuse/volumes: "true"` -- set here via
`fuse_pod_annotations`. (Mountpoint-S3 on EKS needs no such annotation; see `llamacpp_app_mountpoint_s3.py`.)

Prerequisite -- the read-only model PVC (cloud infrastructure, provisioned outside the SDK)
--------------------------------------------------------------------------------------------
`model_delivery="fuse"` mounts a PVC that exposes the dataplane data bucket's `models/`
prefix read-only. On Union GKE this is provisioned by the dataplane helm release (gcsfuse CSI)
as a static PV/PVC, e.g. `union-models-ro`. Each served model is a subdirectory under that
prefix; the app names it via `model_path` (a *relative subpath*, not a remote URI).

Run → Model artifact → FUSE-streamed serve (the full loop)
----------------------------------------------------------

```
python examples/genai/llamacpp/llamacpp_app_gcsfuse.py
```

`__main__` shows the end-to-end loop: (1) a Flyte run (`hf_model`) creates a versioned Model
**artifact**, writing its bytes under the object-store prefix the RO PVC mounts
(`raw_data_path`); (2) the run's artifact URI is stripped of that mounted prefix to derive the
mount-relative subpath; (3) the app streams *exactly that artifact* in place via gcsfuse -- no
download -- so the served weights are the ones this run produced, not a hard-coded path.

Alternatively, deploy the module-scope app directly against a known subpath (no run wiring):

```
flyte deploy examples/genai/llamacpp/llamacpp_app_gcsfuse.py gcsfuse_app
```

Usage is identical to `llamacpp_app.py` (OpenAI-compatible client against `<endpoint>/v1`).
"""

from flyteplugins.llamacpp import LlamaCppAppEnvironment

import flyte
import flyte.app

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
QUANT = "q4_k_m"
ARTIFACT_NAME = "qwen2-5-0-5b-instruct-q4-k-m"

# The object-store prefix the RO model PVC mounts. The prefetch writes the Model artifact
# under this prefix (`raw_data_path`, below), and the app serves a subpath *under the mount*
# -- so this one constant ties "where the run writes the artifact" to "what the FUSE mount
# exposes". Substitute your GCS dataplane data bucket.
MODEL_BUCKET_PREFIX = "gs://<your-dataplane-data-bucket>/models/"

# Where this model lives relative to the mount. Used for the module-scope app (bare
# `flyte deploy`); `__main__` instead *derives* the subpath from the prefetch run's artifact.
MODEL_SUBPATH = "qwen2.5-0.5b-instruct/q4_k_m"

# The read-only, GCS-backed PVC (dataplane helm release). One claim per namespace covers every
# model; each is a subdirectory under the mount.
MODEL_PVC = "union-models-ro"

gcsfuse_app = LlamaCppAppEnvironment(
    name="qwen2-5-0-5b-instruct-gcsfuse",
    model_id="qwen2.5-0.5b-instruct",
    # Lazy gcsfuse mount instead of a download: the weights are read in place from the RO PVC.
    model_delivery="fuse",
    model_pvc=MODEL_PVC,
    model_path=MODEL_SUBPATH,  # relative subpath under the mount, not a remote URI
    # GKE only: the gcsfuse sidecar injector requires this pod annotation.
    fuse_pod_annotations={"gke-gcsfuse/volumes": "true"},
    resources=flyte.Resources(cpu="2", memory="8Gi", gpu="L4:1", disk="10Gi"),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,  # scale to zero after 5 minutes idle; the gcsfuse mount releases clean
    ),
    requires_auth=True,
    extra_args="--ctx-size 8192",
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Run

    flyte.init_from_config()

    # 1. A Flyte run creates the Model artifact. `hf_model` prefetches one quant (allow_patterns
    #    keeps it to the Q4_K_M file) and publishes it as a versioned artifact, writing the bytes
    #    under the object-store prefix the RO PVC mounts (raw_data_path).
    run: Run = flyte.prefetch.hf_model(
        repo=MODEL_REPO,
        artifact_name=ARTIFACT_NAME,
        allow_patterns=[f"*{QUANT}*"],
        raw_data_path=f"{MODEL_BUCKET_PREFIX}{MODEL_SUBPATH}",
        hf_token_key=None,  # public repo: prefetch anonymously
        resources=flyte.Resources(cpu="2", memory="4Gi", disk="10Gi"),
    )
    print(f"Prefetching {MODEL_REPO} ({QUANT}) into the gcsfuse prefix: {run.url}")
    run.wait()

    # 2. Derive the mount-relative subpath from the run's artifact output. The run's output Dir
    #    IS the Model artifact's location; stripping the mounted prefix yields the subpath the RO
    #    PVC exposes -- so the serve streams exactly the artifact this run produced, not a
    #    hard-coded path.
    artifact_uri = run.outputs()[0].path
    subpath = artifact_uri.removeprefix(MODEL_BUCKET_PREFIX).strip("/")
    print(f"Model artifact at {artifact_uri} -> FUSE subpath {subpath!r}")

    # 3. Serve it via a lazy gcsfuse mount (no download): the app reads the artifact in place
    #    from the RO PVC and releases the mount cleanly when it scales to zero.
    app = flyte.serve(gcsfuse_app.clone_with(name=gcsfuse_app.name, model_path=subpath))
    print(f"Deployed llama.cpp gcsfuse app streaming the artifact: {app.url}")
