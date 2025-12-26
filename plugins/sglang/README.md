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
    model="s3://your-bucket/models/your-model",
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
- **OpenAI-Compatible API**: The deployed app exposes an OpenAI-compatible API for chat completions.
- **Auto-scaling**: Configure scaling policies to scale up/down based on traffic.
- **Tensor Parallelism**: Support for distributed inference across multiple GPUs.

## Extra Arguments

You can pass additional arguments to the SGLang server using the `extra_args` parameter:

```python
sglang_app = SGLangAppEnvironment(
    name="my-llm-app",
    model="s3://your-bucket/models/your-model",
    model_id="your-model-id",
    extra_args="--max-model-len 8192 --enforce-eager",
)
```

See the [SGLang server arguments documentation](https://docs.sglang.ai/backend/server_arguments.html) for available options.

