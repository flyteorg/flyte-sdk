# Union llama.cpp Plugin

Serve GGUF models with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` behind Flyte Apps.

This plugin provides the `LlamaCppAppEnvironment` class for deploying quantized (GGUF) LLMs
with an OpenAI-compatible API (under `/v1`) and the built-in llama.cpp Web UI. llama.cpp shines
where vLLM and SGLang don't fit: quantized GGUF weights, partial CPU offload of models larger
than VRAM, and CPU-only serving.

## Installation

```bash
pip install --pre flyteplugins-llamacpp
```

## Usage

```python
import flyte
import flyte.app
from flyteplugins.llamacpp import LlamaCppAppEnvironment

llama_app = LlamaCppAppEnvironment(
    name="my-llm-app",
    # A directory (or direct path) of GGUF weights in object storage...
    model_path="s3://your-bucket/models/your-model-gguf",
    model_id="your-model-id",
    resources=flyte.Resources(cpu="4", memory="32Gi", gpu="L40s:1", disk="100Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(llama_app)
    print(f"Deployed llama.cpp app: {app.url}")
```

`model_path` accepts a remote directory or file path, a `RunOutput` (e.g. from a prefetch task
that downloaded the GGUF), or an `ArtifactValue`. The weights are downloaded into the container
and the served `.gguf` is located at startup; for sharded models the `-00001-of-` shard is
selected and llama-server discovers the rest.

Alternatively, point directly at a Hugging Face GGUF repo (with an optional quant tag) and let
llama-server download it at startup:

```python
llama_app = LlamaCppAppEnvironment(
    name="gemma-app",
    model_hf_path="ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
    model_id="gemma-3-4b-it",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1", disk="50Gi"),
)
```

## The default image

llama.cpp ships no GPU pip wheel, so the default image compiles `llama-server` from source with
CUDA enabled (plus the embedded Web UI). The default targets compute capability 8.9 (L4/L40S);
use `build_llama_cpp_image` to target other GPUs, pin a llama.cpp release for reproducible
builds, or build a CPU-only image:

```python
from flyteplugins.llamacpp import LlamaCppAppEnvironment, build_llama_cpp_image

llama_app = LlamaCppAppEnvironment(
    name="my-llm-app",
    image=build_llama_cpp_image(
        cuda_arch="80;86;89;90",  # fat binary: A100, A10, L4/L40S, H100
        ref="b6148",              # pin a llama.cpp release tag
    ),
    ...
)
```

`build_llama_cpp_image(cuda=False)` produces a CPU-only image for serving small quantized
models without a GPU.

## Speculative decoding

Point `draft_model_path` (object storage, `RunOutput`, or `ArtifactValue`) or
`draft_model_hf_path` at a small draft GGUF and it is passed to llama-server as
`--model-draft` / `--hf-repo-draft`. Tune the speculation via `extra_args`:

```python
llama_app = LlamaCppAppEnvironment(
    name="qwen3-spec",
    model_path="s3://your-bucket/models/qwen3-32b-gguf",
    model_id="qwen3-32b",
    draft_model_hf_path="ggml-org/Qwen3-0.6B-GGUF:Q8_0",
    extra_args="--draft-max 16 --draft-min 1 --gpu-layers-draft 99",
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu="L40s:1", disk="120Gi"),
)
```

## Extra arguments

`extra_args` is appended to `llama-server`, as either a string or a list:

```python
llama_app = LlamaCppAppEnvironment(
    name="my-llm-app",
    model_path="s3://your-bucket/models/your-model-gguf",
    model_id="your-model-id",
    extra_args="--ctx-size 32768 --parallel 4 --jinja",
)
```

Useful flags: `--ctx-size` (context length), `--parallel` (concurrent request slots),
`--jinja` (enable the model's chat template, needed for tool calling), `--n-gpu-layers`
(limit GPU offload for models larger than VRAM; recent llama.cpp offloads everything by
default), `--cache-type-k/--cache-type-v` (quantized KV cache), `--flash-attn`.

Arguments are quoted before they reach the server, so values containing spaces or JSON survive
intact. Arguments of the form `$MY_VAR` are left unquoted so that Flyte still expands them from
the app's environment.

Run `llama-server --help` or see the
[llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
for all options.
