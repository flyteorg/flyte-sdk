"""
A simple vLLM app example deploying the smallest Qwen3 model.

This example shows how to use the VLLMAppEnvironment to serve a model using vLLM.

Deploy
------

Deploy this app using the Flyte CLI:

```
flyte deploy examples/genai/vllm/vllm_app.py vllm_app
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

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app
from flyte._image import DIST_FOLDER, PythonWheels

# Define the vLLM app environment for the smallest Qwen3 model
vllm_app = VLLMAppEnvironment(
    name="qwen3-0-6b-vllm",
    model_hf_path="Qwen/Qwen3-0.6B",
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:4", disk="10Gi"),
    image=(
        flyte.Image.from_debian_base(name="vllm-app-image", install_flyte=False)
        .with_pip_packages("flashinfer-python", "flashinfer-cubin")
        .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu129")
        .with_pip_packages("vllm==0.13.0")
        # .with_local_v2()
        # NOTE: due to a dependency conflict, the vllm flyte plugin needs to be installed as a separate layer:
        # Run the following command to build the wheel:
        # `rm -rf ./dist-plugins && uv run python -m build --wheel --installer uv --outdir ./dist-plugins plugins/vllm`
        # Once a release of the plugin is out, you can installed it via `with_pip_packages("flyteplugins-vllm")`
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=DIST_FOLDER.parent / "dist-plugins", package_name="flyteplugins-vllm", pre=True
            )
        )
        .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte", pre=True))
    ),
    stream_model=True,  # Stream model directly from blob store to GPU
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # (min_replicas, max_replicas)
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    requires_auth=False,
    extra_args=["--max-model-len 8192"],  # Limit context length for smaller GPU memory
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Run

    flyte.init_from_config()

    # prefetch the Qwen3-0.6B model into flyte object store
    run: Run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-0.6B")
    run.wait()
    print(run.url)

    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
            model_hf_path=None,
        )
    )
    print(f"Deployed vLLM app: {app.url}")
