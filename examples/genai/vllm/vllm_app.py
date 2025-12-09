"""
A simple vLLM app example deploying the smallest Qwen3 model.

This example shows how to use the VLLMAppEnvironment to serve a model using vLLM.

Prerequisites
-------------

1. Cache the model to a blob store location that your Flyte deployment can access.
   Run the cache_model.py task to download the model from Hugging Face:

   ```
   flyte run examples/genai/vllm/cache_model.py main \
       --model_id Qwen/Qwen3-0.6B
   ```

   The output will provide the model path (e.g., s3://your-bucket/path/to/model).

2. Set the MODEL_PATH environment variable to point to your cached model:

   ```
   export MODEL_PATH=<model-path-from-cache-output>
   ```

Deploy
------

Deploy this app using the Flyte CLI:

```
flyte deploy examples/genai/vllm/vllm_app.py vllm_app
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
import flyte.app
from flyte._image import PythonWheels, DIST_FOLDER
from flyteplugins.vllm import VLLMAppEnvironment

# Model path should point to a blob store location containing the model weights.
# This can be set via environment variable or hardcoded for testing.
MODEL_PATH = os.environ.get("MODEL_PATH", "s3://your-bucket/models/qwen3-0.6b")

# Define the vLLM app environment for the smallest Qwen3 model
vllm_app = VLLMAppEnvironment(
    name="qwen3-0-6b-vllm",
    model=MODEL_PATH,
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:4", disk="10Gi"),
    image=(
        flyte.Image.from_debian_base(name="vllm-app-image", python_version=(3, 12), install_flyte=False)
        .with_pip_packages("vllm==0.11.0")
        # .with_local_v2()
        .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte", pre=True))
        # # NOTE: due to a dependency conflict, the vllm flyte plugin needs to be installed from source as a separate layer:
        # # Run the following command to build the wheel:
        # # `uv run python -m build --wheel --installer uv --outdir ./dist-plugins plugins/vllm`
        # # Once a release of the plugin is out, you can installed it via `with_pip_packages("flyteplugins-vllm")`
        .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER.parent / "dist-plugins", package_name="flyteplugins-vllm", pre=True))
    ),
    stream_model=True,  # Stream model directly from blob store to GPU
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # (min_replicas, max_replicas)
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    requires_auth=False,
    extra_args=["--max-model-len 8192", "--enforce-eager"],  # Limit context length for smaller GPU memory
    links=[flyte.app.Link(path="/docs", title="Swagger Docs UI", is_relative=True)],
)


if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(vllm_app)
    print(f"Deployed vLLM app: {app.url}")
