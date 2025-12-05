"""
A simple vLLM app example deploying the smallest Qwen3 model.

This example shows how to use the VLLMAppEnvironment to serve a model using vLLM.

Prerequisites
-------------

1. Cache the model to a blob store location that your Flyte deployment can access.
   For example, you can use the Union CLI to cache a model from Hugging Face:

   ```
   flyte run cache_model.py cache_model_from_hf \
       --model_id Qwen/Qwen3-0.6B \
       --output_path s3://your-bucket/models/qwen3-0.6b
   ```

   Or manually upload the model weights to your blob store.

2. Set the MODEL_PATH environment variable to point to your cached model:

   ```
   export MODEL_PATH=s3://your-bucket/models/qwen3-0.6b
   ```

Deploy
------

Deploy this app using the Flyte CLI:

```
flyte deploy examples/genai/vllm/simple.app.py
```

Usage
-----

Once deployed, you can interact with the model using the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="<your-app-endpoint>/v1",
    api_key="<your-api-key>",
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
)
print(response.choices[0].message.content)
```
"""

import os

import flyte
from flyte.app.extras import VLLMAppEnvironment

# Model path should point to a blob store location containing the model weights.
# This can be set via environment variable or hardcoded for testing.
MODEL_PATH = os.environ.get("MODEL_PATH", "s3://your-bucket/models/qwen3-0.6b")

# Define the vLLM app environment for the smallest Qwen3 model
vllm_app = VLLMAppEnvironment(
    name="qwen3-0.6b-vllm",
    model=MODEL_PATH,
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="1"),
    stream_model=True,  # Stream model directly from blob store to GPU
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # (min_replicas, max_replicas)
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    requires_auth=True,
    extra_args="--max-model-len 8192",  # Limit context length for smaller GPU memory
)


if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(vllm_app)
    print(f"Deployed vLLM app: {deployments[0]}")

