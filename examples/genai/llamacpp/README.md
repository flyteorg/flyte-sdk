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

The default model is [`Qwen/Qwen2.5-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
at `q4_k_m` (~0.4 GB) — small enough to iterate on quickly.

## Delivery modes: download vs. lazy FUSE mount

The same prefetched model can reach the server two ways, set by `model_delivery`:

- **`"download"`** (default, [`llamacpp_app.py`](llamacpp_app.py)) — the bound `ArtifactValue`
  is copied into the pod's local disk before `llama-server` starts. Simple; the whole GGUF
  lands on the node's ephemeral disk and the copy is on the cold-start path.
- **`"fuse"`** — the weights are read **in place** from a read-only, object-store-backed PVC.
  Nothing is copied to local disk, first touch is lazy, and the mount releases cleanly on
  scale-to-zero — so scale-from-zero is bounded by GPU node cold-start, not by re-downloading
  the model. A `PersistentVolumeClaim` is Knative-friendly and, unlike a node device-plugin
  (JuiceFS / Union Volume), does not block scale-to-zero. The PVC is cloud infrastructure
  provisioned outside the SDK (the dataplane helm release); the app names a *relative subpath*
  under it via `model_path`. One example per CSI driver:
  - [`llamacpp_app_gcsfuse.py`](llamacpp_app_gcsfuse.py) — GKE / gcsfuse (needs the
    `gke-gcsfuse/volumes: "true"` pod annotation).
  - [`llamacpp_app_mountpoint_s3.py`](llamacpp_app_mountpoint_s3.py) — EKS / Mountpoint-S3
    (mounts the static PV directly; no annotation).

| Delivery | Local disk | First touch | Scale-to-zero |
|---|---|---|---|
| `download` | full GGUF copied | after full download | clean, re-downloads on wake |
| `fuse` | none | lazy (~20-25 s for ~18 GB) | clean, no re-download |

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
