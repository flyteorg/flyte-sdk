# Flyte SGLang Plugin

Serve large language models using SGLang with Flyte Apps.

This plugin provides the `SGLangAppEnvironment` class for deploying and serving LLMs using [SGLang](https://docs.sglang.ai/).

## Installation

```bash
pip install --pre flyteplugins-sglang
```

## Usage

```python
import flyte
import flyte.app
from flyteplugins.sglang import SGLangAppEnvironment

# Define the SGLang app environment
sglang_app = SGLangAppEnvironment(
    name="my-llm-app",
    model_path="s3://your-bucket/models/your-model",
    model_id="your-model-id",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:1"),
    stream_model=True,  # Stream model directly from blob store to GPU
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,
    ),
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(sglang_app)
    print(f"Deployed SGLang app: {app.url}")
```

## Features

- **Streaming Model Loading**: Stream model weights directly from object storage to GPU memory, reducing startup time and disk requirements.
- **Speculative Decoding**: Serve a target model alongside a draft model (speculator) to shorten decode.
- **Cache-Aware Routing**: Run data-parallel workers behind SGLang's router so requests land on the worker that already holds their prefix.
- **OpenAI-Compatible API**: The deployed app exposes an OpenAI-compatible API for chat completions.
- **Auto-scaling**: Configure scaling policies to scale up/down based on traffic.
- **Tensor Parallelism**: Support for distributed inference across multiple GPUs.

## Speculative decoding

Set `speculative_config` to turn on
[speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html). Each
key becomes a `--speculative-<key>` server argument. When the speculator has its own weights,
point `draft_model_path` (object storage, `RunOutput` or `ArtifactValue`) or `draft_model_hf_path`
at them and the plugin mounts them next to the target model and fills in
`--speculative-draft-model-path`:

```python
sglang_app = SGLangAppEnvironment(
    name="qwen3-8b-spec",
    model_path="s3://your-bucket/models/qwen3-8b",
    model_id="qwen3-8b",
    draft_model_path="s3://your-bucket/models/qwen3-8b-eagle3",
    speculative_config={"algorithm": "EAGLE3"},
    extra_args=["--mem-fraction-static", "0.75", "--trust-remote-code", "--enable-metrics"],
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu="L40s:1", disk="120Gi"),
)
```

Supported algorithms include `EAGLE3`, `EAGLE`, `DFLASH`, `STANDALONE` (an independent small
model as the drafter) and `NGRAM` (no draft model at all).

Three caveats worth knowing up front:

- **Let SGLang pick `num_steps`, `eagle_topk` and `num_draft_tokens` unless you have measured
  otherwise.** Its defaults are model-family dependent — Llama/Grok want `topk=4`, everything else
  `topk=1` — so a triple copied from a Llama example can make a Qwen target *slower* than no
  speculation at all.
- **Streaming is disabled whenever a draft model is configured.** The Flyte loader integration is
  installed process-wide for a single set of weights, so both models are downloaded to the
  container instead. Size `disk` for target + draft + a margin.
- **Measure acceptance length, not just tokens/sec.** Add `--enable-metrics` and read acceptance
  off `/metrics`; a value near 1 means the draft model is wrong for your target.

## Cache-aware routing

`router=True` serves the app through
[SGLang's router](https://docs.sglang.io/advanced_features/router.html), which runs data-parallel
workers in one process and sends each request to the worker most likely to already hold its
prefix. Worker count and policy are ordinary server arguments:

```python
sglang_app = SGLangAppEnvironment(
    name="my-llm-app",
    model_path="s3://your-bucket/models/your-model",
    model_id="your-model-id",
    router=True,
    extra_args=["--dp-size", "4", "--tp-size", "1", "--router-policy", "cache_aware"],
    resources=flyte.Resources(cpu="32", memory="200Gi", gpu="L40s:4", disk="120Gi", shm="auto"),
    scaling=flyte.app.Scaling(replicas=(1, 1)),
)
```

Keeping routing inside the pod is what makes prefix affinity expressible at all: Flyte apps have
no session affinity and no per-replica address, so the router has to sit in front of its own
workers rather than in front of replicas. Note that the KV cache is per replica and dies with it,
so scaling to zero throws away exactly what the router is building — prefer `replicas=(1, 1)`.

Streaming is disabled in router mode: the router launches its workers as separate processes that
never load the Flyte model loader.

## Extra arguments

`extra_args` is appended to the SGLang server command, as either a string or a list:

```python
sglang_app = SGLangAppEnvironment(
    name="my-llm-app",
    model_path="s3://your-bucket/models/your-model",
    model_id="your-model-id",
    extra_args="--context-length 8192",
)
```

Arguments are quoted before they reach the server, so values containing spaces or JSON survive
intact. Arguments of the form `$MY_VAR` are left unquoted so that Flyte still expands them from
the app's environment. The server binds `0.0.0.0` unless you pass your own `--host`.

See the [SGLang server arguments documentation](https://docs.sglang.ai/backend/server_arguments.html) for available options.
