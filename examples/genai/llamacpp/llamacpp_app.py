"""
Serve a GGUF model with llama.cpp, delivered as a prefetched model **artifact**.

This is the llama.cpp counterpart to `examples/genai/vllm` and `examples/genai/sglang`.
Where those serve safetensors weights with a GPU-streaming loader, llama.cpp serves
quantized **GGUF** weights -- the format vLLM and SGLang don't take -- and shines exactly
where they don't fit: quantized models, partial CPU offload of models larger than VRAM,
and CPU-only serving.

The delivery pattern mirrors `sglang_app_artifact.py`: `flyte.prefetch.hf_model` publishes
the model as a versioned artifact, and the app binds it by artifact name -- so the app is
complete at module scope and deployable with a plain `flyte deploy`, no run name to thread.

The one llama.cpp-specific wrinkle is **file selection**. A GGUF repo ships many
quantizations (q3, q4, q5, q6, q8, ...) and you want exactly one. `hf_model`'s
`allow_patterns` prefetches just that quant instead of the whole repo, and the stored
subset is what the app serves.

Step 1 -- prefetch one quant (publishes the artifact)
-----------------------------------------------------

```
python examples/genai/llamacpp/llamacpp_app.py
```

`allow_patterns=["*q4_k_m*"]` stores only the Q4_K_M GGUF (~0.4 GB) out of a repo that also
ships q3/q5/q6/q8, so the artifact is the single file the server loads. Inspect it with:

```
flyte get artifact qwen2-5-0-5b-instruct-q4-k-m
```

The repo is public, so `hf_token_key=None` prefetches anonymously -- no HF_TOKEN secret.

Step 2 -- deploy the app
------------------------

```
flyte deploy examples/genai/llamacpp/llamacpp_app.py llamacpp_app
```

`ArtifactValue` resolves at deploy time and pins the app to the artifact version, so a later
re-prefetch does not swap the weights under a running app. Redeploy to move forward.

Usage
-----

```python
from openai import OpenAI

client = OpenAI(base_url="<your-app-endpoint>/v1", api_key="<your-api-key>")

response = client.chat.completions.create(
    model="qwen2.5-0.5b-instruct",
    messages=[{"role": "user", "content": "Write a one-line hello in Python."}],
)
print(response.choices[0].message.content)
```
"""

from flyteplugins.llamacpp import LlamaCppAppEnvironment

import flyte
import flyte.app

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
QUANT = "q4_k_m"

# hf_model requires an artifact name of [alnum_-] only, and a GGUF repo holds many
# quants at one commit -- so the quant is encoded into the artifact name to keep each
# prefetched quant a distinct, addressable artifact.
ARTIFACT_NAME = "qwen2-5-0-5b-instruct-q4-k-m"

llamacpp_app = LlamaCppAppEnvironment(
    name="qwen2-5-0-5b-instruct-llamacpp",
    model_id="qwen2.5-0.5b-instruct",
    # Bind the prefetched artifact, not a run -- deployable with a bare `flyte deploy`.
    model_path=flyte.app.ArtifactValue(name=ARTIFACT_NAME, type="directory"),
    # A 0.5B Q4_K_M GGUF runs comfortably on a single L4; for CPU-only serving, drop the
    # gpu and build a CPU image with `build_llama_cpp_image(cuda=False)` (see the README).
    resources=flyte.Resources(cpu="2", memory="8Gi", gpu="L4:1", disk="10Gi"),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,  # scale to zero after 5 minutes idle
    ),
    requires_auth=True,
    extra_args="--ctx-size 8192",
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Run

    flyte.init_from_config()

    # Prefetch ONE quant out of the multi-quant GGUF repo. Without allow_patterns this
    # would pull every quant in the repo; with it, only the Q4_K_M file is stored -- and
    # published as the artifact the app binds above.
    run: Run = flyte.prefetch.hf_model(
        repo=MODEL_REPO,
        artifact_name=ARTIFACT_NAME,
        allow_patterns=[f"*{QUANT}*"],
        hf_token_key=None,  # public repo: prefetch anonymously
        resources=flyte.Resources(cpu="2", memory="4Gi", disk="10Gi"),
    )
    print(f"Prefetching {MODEL_REPO} ({QUANT}): {run.url}")
    run.wait()

    # Nothing needs to be passed from the run to the app -- `llamacpp_app` already names
    # the artifact, and deploy resolves it.
    app = flyte.serve(llamacpp_app)
    print(f"Deployed llama.cpp app: {app.url}")
