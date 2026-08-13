"""
Serve a prefetched HuggingFace model from its **artifact**, using SGLang.

This is the artifact-bound counterpart to `sglang_app.py`, and the SGLang twin of
`examples/genai/vllm/vllm_app_artifact.py`. Instead of pointing the app at a
specific run (`flyte.app.RunOutput(run_name=...)`), the app declares the model it
wants by *artifact name*:

```python
model_path=flyte.app.ArtifactValue(name="SmolLM2-135M-Instruct", type="directory")
```

Why that matters: `flyte.prefetch.hf_model()` publishes a model artifact whose
name is the repo tail and whose version is the HuggingFace commit id. Binding to
the artifact name means the app definition is complete at module scope — it is
deployable with a plain `flyte deploy`, with no run name to thread through — and
the resolved artifact version is recorded, so the platform links this app to the
exact model version it is serving.

The model is `HuggingFaceTB/SmolLM2-135M-Instruct` (135M params, ~270MB bf16),
the smallest checkpoint that satisfies both ends of this pipeline:

- prefetch requires **safetensors** weights (the streaming serving loader reads
  nothing else), which rules out e.g. `facebook/opt-125m`;
- SGLang requires a resolvable `architectures` entry in config.json, which rules
  out the sub-1MB `hf-internal-testing/tiny-random-gpt2`.

SmolLM2-135M is `LlamaForCausalLM` with head_dim 64 and a chat template, so it
loads on any attention backend and answers on the OpenAI-compatible route.

Step 1 — prefetch the model (publishes the artifact)
----------------------------------------------------

```
python examples/genai/sglang/sglang_app_artifact.py
```

The repo is public, so `hf_token_key=None` prefetches anonymously — no HF_TOKEN
secret required. Inspect what was published with:

```
flyte get artifact SmolLM2-135M-Instruct
```

The artifact is engine-agnostic: the same prefetch feeds this app and the vLLM
one. Only unsharded prefetches are portable that way — a `shard_config=` prefetch
writes per-rank files for one engine at one tensor-parallel degree.

Step 2 — deploy the app
-----------------------

```
flyte deploy examples/genai/sglang/sglang_app_artifact.py smollm2_app
```

`ArtifactValue` is resolved at deploy time: `version=None` picks the latest
version and pins the app to it, so a later re-prefetch of a new HF commit does
not silently swap the weights under a running app. Redeploy to move forward.

Usage
-----

```python
from openai import OpenAI

client = OpenAI(base_url="<your-app-endpoint>/v1", api_key="<your-api-key>")

response = client.chat.completions.create(
    model="smollm2-135m",
    messages=[{"role": "user", "content": "Explain what a Flyte artifact is, briefly."}],
)
print(response.choices[0].message.content)
```
"""

from flyteplugins.sglang import SGLangAppEnvironment

import flyte
import flyte.app

MODEL_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"

# The artifact name that `flyte.prefetch.hf_model` publishes by default: the repo
# tail, with '.' replaced by '-'. Pass `artifact_name=` to hf_model to override it.
ARTIFACT_NAME = "SmolLM2-135M-Instruct"

image = (
    flyte.Image.from_debian_base(name="sglang-app-image", install_flyte=False)
    .with_apt_packages("libnuma-dev", "wget", "curl", "openssl", "pkg-config", "libssl-dev", "build-essential")
    .with_commands(
        [
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
            "apt-get install -y cuda-toolkit-12-8",
        ]
    )
    .with_commands(["curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . $HOME/.cargo/env"])
    .with_env_vars({"CUDA_HOME": "/usr/local/cuda-12.8", "PATH": "/root/.cargo/bin:/usr/local/cuda-12.8/bin:$PATH"})
    .with_pip_packages("flashinfer-python", "flashinfer-cubin")
    .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu128")
    .with_pip_packages("sglang==0.5.7")
    .with_pip_packages("flyteplugins-sglang")
)

smollm2_app = SGLangAppEnvironment(
    name="smollm2-135m-sglang",
    model_id="smollm2-135m",
    # The whole point of this example: bind to the artifact, not to a run.
    # Add `version="<hf-commit-sha>"` to pin a specific checkpoint instead of
    # resolving the latest at deploy time.
    model_path=flyte.app.ArtifactValue(name=ARTIFACT_NAME, type="directory"),
    # bf16 weights, so the GPU needs compute capability >= 8.0 — L4 (Ada) is the
    # cheapest that qualifies. T4 is Turing and would fall over on bfloat16.
    resources=flyte.Resources(cpu="2", memory="8Gi", gpu="L4:1", disk="10Gi"),
    image=image,
    # Stream the safetensors straight from the blob store into GPU memory rather
    # than staging them on local disk first.
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,
    ),
    requires_auth=True,
    # SGLang spells this `--context-length`; `--max-model-len` is the vLLM flag
    # and SGLang's launcher rejects it. Passed pre-split, one token per element.
    extra_args=["--context-length", "8192"],
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Run

    flyte.init_from_config()

    # Prefetch the model into the Flyte object store. On success this publishes a
    # model artifact named ARTIFACT_NAME, versioned by the HuggingFace commit id,
    # with the repo README attached as the model card.
    run: Run = flyte.prefetch.hf_model(
        repo=MODEL_REPO,
        hf_token_key=None,  # public repo: prefetch anonymously
        resources=flyte.Resources(cpu="2", memory="4Gi", disk="10Gi"),
    )
    print(f"Prefetching {MODEL_REPO}: {run.url}")
    run.wait()

    # Nothing needs to be passed from the run to the app — `smollm2_app` already
    # names the artifact, and deploy resolves it.
    app = flyte.serve(smollm2_app)
    print(f"Deployed SGLang app: {app.url}")
