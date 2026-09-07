# Union vLLM Plugin

Serve large language models using vLLM with Flyte Apps.

This plugin provides the `VLLMAppEnvironment` class for deploying and serving LLMs using [vLLM](https://docs.vllm.ai/).

## Installation

```bash
pip install --pre flyteplugins-vllm
```

## Usage

```python
import flyte
import flyte.app
from flyteplugins.vllm import VLLMAppEnvironment

# Define the vLLM app environment
vllm_app = VLLMAppEnvironment(
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
    app = flyte.serve(vllm_app)
    print(f"Deployed vLLM app: {app.url}")
```

## Features

- **Streaming Model Loading**: Stream model weights directly from object storage to GPU memory, reducing startup time and disk requirements.
- **Speculative Decoding**: Serve a target model alongside a draft model (speculator) to shorten decode.
- **OpenAI-Compatible API**: The deployed app exposes an OpenAI-compatible API for chat completions.
- **Auto-scaling**: Configure scaling policies to scale up/down based on traffic.
- **Tensor Parallelism**: Support for distributed inference across multiple GPUs.

## Speculative decoding

Set `speculative_config` to turn on [speculative decoding](https://docs.vllm.ai/en/stable/features/spec_decode/).
It is rendered into vLLM's `--speculative-config` JSON blob. When the speculator has its own
weights, point `draft_model_path` (object storage, `RunOutput` or `ArtifactValue`) or
`draft_model_hf_path` at them and the plugin mounts them next to the target model and fills in
the config's `model` key:

```python
vllm_app = VLLMAppEnvironment(
    name="qwen3-8b-spec",
    model_path="s3://your-bucket/models/qwen3-8b",
    model_id="qwen3-8b",
    draft_model_path="s3://your-bucket/models/qwen3-8b-eagle3",
    speculative_config={"method": "eagle3", "num_speculative_tokens": 3},
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu="L40s:1", disk="120Gi"),
)
```

Methods that need no separate weights work the same way, without a draft model — n-gram is a
useful floor to measure a real speculator against:

```python
speculative_config={"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}
```

Two caveats worth knowing up front:

- **Streaming is disabled whenever a draft model is configured.** The Flyte streaming loader is
  process-wide and describes a single set of weights, so both models are downloaded to the
  container instead. Size `disk` for target + draft + a margin.
- **Measure acceptance length, not just tokens/sec.** A speculator that produces no speedup is
  indistinguishable from a mis-tuned one until you read
  `vllm:spec_decode_num_accepted_tokens_total` / `..._num_drafts_total`. Acceptance ≈ 1 means the
  draft model is wrong for your target.

## Extra arguments

`extra_args` is appended to `vllm serve`, as either a string or a list:

```python
vllm_app = VLLMAppEnvironment(
    name="my-llm-app",
    model_path="s3://your-bucket/models/your-model",
    model_id="your-model-id",
    extra_args="--max-model-len 8192 --quantization fp8",
)
```

Arguments are quoted before they reach the server, so values containing spaces or JSON survive
intact. Arguments of the form `$MY_VAR` are left unquoted so that Flyte still expands them from
the app's environment.

See the [vLLM engine arguments documentation](https://docs.vllm.ai/en/stable/configuration/engine_args)
for available options.
