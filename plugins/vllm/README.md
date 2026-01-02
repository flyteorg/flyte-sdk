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
    app = flyte.serve(vllm_app)
    print(f"Deployed vLLM app: {app.url}")
```

## Features

- **Streaming Model Loading**: Stream model weights directly from object storage to GPU memory, reducing startup time and disk requirements.
- **OpenAI-Compatible API**: The deployed app exposes an OpenAI-compatible API for chat completions.
- **Auto-scaling**: Configure scaling policies to scale up/down based on traffic.
- **Tensor Parallelism**: Support for distributed inference across multiple GPUs.

