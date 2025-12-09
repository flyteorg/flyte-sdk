"""
A simple SGLang app example deploying the smallest Qwen3 model.

This example shows how to use the SGLangAppEnvironment to serve a model using SGLang.

Prerequisites
-------------

1. Cache the model to a blob store location that your Flyte deployment can access.
   Run the cache_model.py task to download the model from Hugging Face:

   ```
   flyte run examples/genai/sglang/cache_model.py main \
       --model_id Qwen/Qwen3-0.6B
   ```

   The output will provide the model path (e.g., s3://your-bucket/path/to/model).

Deploy
------

Deploy this app using the Flyte CLI:

```
flyte deploy examples/genai/sglang/sglang_app.py sglang_app
```

Note that `model=flyte.app.RunOutput(run_name="cache-model-env", task_name="main")`
is used to specify the model to use. It will automatically materialize the correct
model path from the latest run of the `cache-model-env.main` task.

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

from flyteplugins.sglang import SGLangAppEnvironment

import flyte
import flyte.app
import flyte.io
from flyte._image import DIST_FOLDER, PythonWheels

# Define the SGLang app environment for the smallest Qwen3 model
sglang_app = SGLangAppEnvironment(
    name="qwen3-0-6b-sglang",
    model=flyte.app.RunOutput(type=flyte.io.Dir, task_name="cache_model_env.main"),
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:4", disk="10Gi"),
    image=(
        flyte.Image.from_debian_base(name="sglang-app-image", python_version=(3, 12), install_flyte=False)
        .with_pip_packages("sglang[all]==0.4.8")
        # .with_local_v2()
        .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte", pre=True))
        # NOTE: due to a dependency conflict, the sglang flyte plugin needs to be installed as a separate layer:
        # Run the following command to build the wheel:
        # `uv run python -m build --wheel --installer uv --outdir ./dist-plugins plugins/sglang`
        # Once a release of the plugin is out, you can install it via `with_pip_packages("flyteplugins-sglang")`
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=DIST_FOLDER.parent / "dist-plugins", package_name="flyteplugins-sglang", pre=True
            )
        )
    ),
    stream_model=True,  # Stream model directly from blob store to GPU
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # (min_replicas, max_replicas)
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    requires_auth=False,
    extra_args=["--max-model-len", "8192", "--enforce-eager"],  # Limit context length for smaller GPU memory
)


if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(sglang_app)
    print(f"Deployed SGLang app: {app.url}")

