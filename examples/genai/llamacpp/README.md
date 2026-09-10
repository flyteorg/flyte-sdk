# llama.cpp serving

Serve quantized **GGUF** models with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-server` behind a Flyte App, with an OpenAI-compatible endpoint at `/v1`.

This is the GGUF counterpart to [`../vllm`](../vllm) and [`../sglang`](../sglang). Those serve
safetensors weights and stream them straight to the GPU; llama.cpp serves the quantized GGUF
format they don't take, and runs where they don't fit: quantized weights, partial CPU offload
of models larger than VRAM, and CPU-only serving. It builds on the `flyteplugins.llamacpp`
plugin — the example is thin: prefetch → artifact → serve.

## The two levers this example shows

**1. Object-store model delivery as a versioned artifact.** `flyte.prefetch.hf_model`
downloads the weights once to blob storage and publishes them as a model **artifact**
(versioned by the HuggingFace commit). The app binds the artifact by name with
`ArtifactValue`, so it is complete at module scope and deploys with a bare `flyte deploy` —
no run name to thread. The app scales to zero when idle and remounts the same weights on the
next request.

**2. File selection for GGUF.** A GGUF repo ships many quantizations at one commit; you serve
exactly one. `hf_model(..., allow_patterns=["*q4_k_m*"])` prefetches only that quant instead
of the whole repo, and records the selected pattern in the artifact metadata so the stored
subset is identifiable. Pull a different quant by changing `QUANT` — each is published as its
own artifact.

## Run it

```bash
# 1. Prefetch one quant and publish the artifact
python examples/genai/llamacpp/llamacpp_app.py

# 2. Deploy the app (resolves the artifact at deploy time)
flyte deploy examples/genai/llamacpp/llamacpp_app.py llamacpp_app

# 3. Call it
python examples/genai/llamacpp/client.py --endpoint <app-endpoint> --api_key <api-key>
```

[`llamacpp_app.py`](llamacpp_app.py) (the download example) defaults to
[`Qwen/Qwen2.5-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) at
`q4_k_m` (~0.4 GB) — small enough to iterate on quickly. The FUSE example
([`llamacpp_app_fuse.py`](llamacpp_app_fuse.py)) instead defaults to a real GPU-class target,
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at `Q8_0` (~27 GB) on
2× L4, and every field (`LLAMACPP_MODEL_REPO`, `LLAMACPP_QUANT`, `LLAMACPP_GPU`, …) is env-overridable
— point it at the small model for quick iteration, or at another cloud's accelerator
(`LLAMACPP_GPU=L40S:1` on AWS, `A10:2` on Azure), without editing.

## Delivery modes: download vs. lazy FUSE mount

The same prefetched model can reach the server two ways, set by `model_delivery`:

- **`"download"`** (default, [`llamacpp_app.py`](llamacpp_app.py)) — the bound `ArtifactValue`
  is copied into the pod's local disk before `llama-server` starts. Simple; the whole GGUF
  lands on the node's ephemeral disk and the copy is on the cold-start path.
- **`"fuse"`** — the weights are read **in place** from a read-only, object-store-backed PVC.
  Nothing is copied to local disk, first touch is lazy, and the mount releases cleanly on
  scale-to-zero — so scale-from-zero is bounded by GPU node cold-start, not by re-downloading
  the model. A `PersistentVolumeClaim` is Knative-friendly and, unlike a node device-plugin
  (JuiceFS / Union Volume), does not block scale-to-zero. The PVC (over the data-bucket root) is
  cloud infrastructure provisioned outside the SDK (the dataplane helm release); the app binds a
  Model artifact via `model_path=ArtifactValue(...)` and it streams in place. One example, both
  CSI backends documented in comments:
  - [`llamacpp_app_fuse.py`](llamacpp_app_fuse.py) — gcsfuse (GKE; needs the
    `gke-gcsfuse/volumes: "true"` pod annotation) and Mountpoint-S3 (EKS; mounts the static PV
    directly, no annotation).

| Delivery | Local disk | First touch | Scale-to-zero |
|---|---|---|---|
| `download` | full GGUF copied | after full download | clean, re-downloads on wake |
| `fuse` | none | lazy (~20-25 s for ~18 GB) | clean, no re-download |

### Prerequisite for `fuse`: the read-only model PVC

`fuse` mode mounts a **pre-provisioned, read-only PVC** that exposes the **root of the dataplane
data bucket** — the same bucket a Model artifact materializes into — so an artifact at
`<scheme>://<data-bucket>/<key>` is read in place at `<mount>/<key>`. No bucket prefix, no subpath.

A valid claim must already exist, and you name it with `model_pvc`. On a Union-managed dataplane it
is provisioned for you — the helm release creates it (`flyte-metadata-ro` by default) and also
exports the name as `FLYTE_MODEL_PVC`. You author the manifests below only on a **self-managed
cluster** where the platform does not provision the claim.

Two things are invariant across both backends: the claim is **static** (`storageClassName: ""` —
dynamic provisioning would create a *new* bucket) and **`ReadOnlyMany`**; and the mount is the
**bucket root** (no gcsfuse `only-dir`, no Mountpoint-S3 `prefix`). They differ in exactly one
subtle place — **which field names the bucket** — so the two are not interchangeable:

**gcsfuse (GKE)** — the bucket is `csi.volumeHandle`; `volumeAttributes.bucketName` is **ignored**.
The pod must also carry `gke-gcsfuse/volumes: "true"` (set via `fuse_pod_annotations`), which
triggers the sidecar injector.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: flyte-metadata-ro-pv
spec:
  accessModes: ["ReadOnlyMany"]
  capacity: { storage: 1Gi }        # ignored by gcsfuse; a required placeholder
  storageClassName: ""              # static: never dynamically provision (that makes a new bucket)
  mountOptions:                     # bucket ROOT — note there is no `only-dir`
    - implicit-dirs
    - file-cache:max-size-mb:-1
    - file-cache:enable-parallel-downloads:true
    - metadata-cache:ttl-secs:-1
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeHandle: <GCS_DATA_BUCKET>  # <-- the bucket goes HERE (volumeAttributes.bucketName is ignored)
    readOnly: true
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: flyte-metadata-ro
  namespace: <project>-<domain>      # the app's namespace, e.g. "development"
spec:
  accessModes: ["ReadOnlyMany"]
  storageClassName: ""
  volumeName: flyte-metadata-ro-pv
  resources: { requests: { storage: 1Gi } }
```

**Mountpoint-S3 (EKS)** — the reverse: `csi.volumeHandle` is any name unique across PVs, and the
bucket is `volumeAttributes.bucketName`. No pod annotation is needed — drop `fuse_pod_annotations`.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: flyte-metadata-ro-pv
spec:
  accessModes: ["ReadOnlyMany"]
  capacity: { storage: 1Gi }        # ignored by Mountpoint-S3; a placeholder
  storageClassName: ""
  mountOptions:                     # bucket ROOT — note there is no `prefix`
    - region <AWS_REGION>
    - read-only
    - allow-other
  csi:
    driver: s3.csi.aws.com
    volumeHandle: flyte-metadata-ro-pv   # any value unique across PVs — NOT the bucket
    volumeAttributes:
      bucketName: <S3_DATA_BUCKET>       # <-- the bucket goes HERE
      authenticationSource: pod
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: flyte-metadata-ro
  namespace: <project>-<domain>
spec:
  accessModes: ["ReadOnlyMany"]
  storageClassName: ""
  volumeName: flyte-metadata-ro-pv
  resources: { requests: { storage: 1Gi } }
```

Then point the app at the claim (or rely on `FLYTE_MODEL_PVC`):

```python
LlamaCppAppEnvironment(..., model_delivery="fuse", model_pvc="flyte-metadata-ro")
```

## Variations

- **CPU-only serving.** Drop `gpu` from `resources` and pass a CPU image:
  ```python
  from flyteplugins.llamacpp import LlamaCppAppEnvironment, build_llama_cpp_image
  llamacpp_app = LlamaCppAppEnvironment(..., image=build_llama_cpp_image(cuda=False))
  ```
- **A different quant or model.** Change `QUANT` / `MODEL_REPO` and the `allow_patterns` glob;
  bump `resources` for larger weights.
- **Serving tuning.** `extra_args` is appended to `llama-server` (e.g. `--ctx-size`, `--parallel`,
  `--jinja` for tool-calling, `--flash-attn`). See the
  [llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
- **Speculative decoding.** Point `draft_model_hf_path` at a small draft GGUF (see the plugin README).
- **Serving shape: task-pod sidecar.** Besides the standalone scale-to-zero App above, you can
  run llama.cpp as a **native sidecar in a Flyte task pod** for batch/pipeline inference against
  a co-located model — see [`llamacpp_sidecar.py`](llamacpp_sidecar.py). It serves a Model
  **artifact over the same object-store FUSE mount** as the fuse App (the sidecar starts before
  the primary, so the weights must be present at startup — a mounted artifact fits, a
  task-input download does not), and builds the server command with the plugin's
  `build_fserve_command` (the same argv the App runs), so both shapes stay in lockstep. Needs
  the same read-only model PVC prerequisite as the fuse App. It runs in two shapes via one
  `--reuse` flag: an **ephemeral pod** (default, fresh per run) or a **reusable actor**
  (`flyte.ReusePolicy` — a warm pod keeps the loaded model across runs; only the first call
  pays the cold-start). Both are injected at submit via `chat.override(pod_template=..., reusable=...)`.
  Env-configurable like the fuse App (`LLAMACPP_*`); set `LLAMACPP_GPU` (e.g. `L4:1`) to put a
  GPU on the sidecar container (request==limit) with a CUDA image and `--n-gpu-layers` offload.
- **Delivery mode: Union Volume (JuiceFS).** A third way to reach the weights —
  [`llamacpp_app_union_fuse.py`](llamacpp_app_union_fuse.py) — alongside `download` and the
  object-store RO-PVC `fuse`. It prefetches the model as an artifact, builds a **Union Volume**
  (`flyteplugins.union.io` — a JuiceFS POSIX fs over object storage, immutable chunks + a
  metadata index whose `locator` rides the Flyte literal system), then serves by mounting the
  volume **read-only** via the **union device-plugin** FUSE (`PodTemplate.allow_fuse()` —
  unprivileged `CAP_SYS_ADMIN` + the `smarter-devices/fuse` extended resource, Knative-friendly).
  One built volume fans out read-only to many replicas with a JuiceFS local cache; the tradeoff
  vs the RO-PVC fuse App is it does **not** cleanly scale to zero (the JuiceFS client subprocess
  pins the pod), so keep ≥1 replica. Needs the dataplane `fuseDevicePlugin` DaemonSet.
