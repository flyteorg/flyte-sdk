"""
A simple SGLang app example deploying the smallest Qwen3 model.

This example shows how to use the SGLangAppEnvironment to serve a model using SGLang.

Deploy
------

Deploy this app using the Flyte CLI:

```
flyte deploy examples/genai/sglang/sglang_app.py sglang_app
```

Note that `model=flyte.app.RunOutput(run_name="cache_model_env", task_name="main")`
is used to specify the model to use. It will automatically materialize the correct
model path from the latest run of the `cache_model_env.main` task.

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

from flyteplugins.sglang import DEFAULT_SGLANG_IMAGE, SGLangAppEnvironment

import flyte
import flyte.app

# The plugin's default image: an SGLang the model loader supports, plus the CUDA toolkit
# matching the CUDA major that SGLang's own wheels pin. Extend it with
# `.clone(name=...).with_pip_packages(...)` when an app needs extra dependencies.
image = DEFAULT_SGLANG_IMAGE

# Define the SGLang app environment for the smallest Qwen3 model
sglang_app = SGLangAppEnvironment(
    name="qwen3-0-6b-sglang",
    model_hf_path="Qwen/Qwen3-0.6B",
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:4", disk="10Gi"),
    image=image,
    stream_model=True,  # Stream model directly from blob store to GPU
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # (min_replicas, max_replicas)
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    requires_auth=True,
    extra_args=["--context-length", "8192"],  # Limit context length for smaller GPU memory
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Run

    flyte.init_from_config()

    # prefetch the Qwen3-0.6B model into flyte object store
    run: Run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-0.6B")
    run.wait()

    app = flyte.serve(
        sglang_app.clone_with(
            name=sglang_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
            model_hf_path=None,
        )
    )
    print(f"Deployed SGLang app: {app.url}")
