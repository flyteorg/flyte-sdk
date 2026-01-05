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
    model="qwen3-14b",
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
from flyte._image import DIST_FOLDER, PythonWheels

image = (
    flyte.Image.from_debian_base(name="sglang-app-image", install_flyte=False)
    .with_apt_packages("libnuma-dev", "wget")
    .with_commands(
        [
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
            "apt-get install -y cuda-toolkit-12-8",
        ]
    )
    .with_pip_packages("flashinfer-python", "flashinfer-cubin")
    .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu128")
    .with_pip_packages("sglang")
    # .with_local_v2()
    # NOTE: call `make dist` to build the flyte wheel
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte", pre=True))
    # NOTE: build the sglang wheel with:
    # `rm -rf ./dist-plugins && uv run python -m build --wheel --installer uv --outdir ./dist-plugins plugins/sglang`
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=DIST_FOLDER.parent / "dist-plugins", package_name="flyteplugins-sglang", pre=True
        )
    )
    .with_env_vars({"CUDA_HOME": "/usr/local/cuda-12.8"})
)

# Define the SGLang app environment for the smallest Qwen3 model
sglang_app = SGLangAppEnvironment(
    name="qwen3-14b-sglang-sharded",
    model_hf_path="Qwen/Qwen3-14B",
    model_id="qwen3-14b",
    resources=flyte.Resources(cpu="36", memory="300Gi", gpu="L40s:4", disk="300Gi", shm="auto"),
    image=image,
    stream_model=True,  # Stream model directly from blob store to GPU
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # (min_replicas, max_replicas)
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    requires_auth=False,
    extra_args=[
        "--context-length",
        "8192",
        "--tensor-parallel-size",
        "4",
        "--mem-fraction-static",
        "0.8",
    ],
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.prefetch import ShardConfig, VLLMShardArgs

    flyte.init_from_config()

    # prefetch the Qwen3-14B model into flyte object store
    run = flyte.prefetch.hf_model(
        repo="Qwen/Qwen3-14B",
        resources=flyte.Resources(cpu="36", memory="300Gi", gpu="L40s:4", disk="300Gi"),
        shard_config=ShardConfig(
            engine="vllm",
            args=VLLMShardArgs(
                tensor_parallel_size=4,
                gpu_memory_utilization=0.9,
                max_model_len=16384,
            ),
        ),
    )
    run.wait()

    app = flyte.serve(
        sglang_app.clone_with(
            name=sglang_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
            model_hf_path=None,
        )
    )
    print(f"Deployed SGLang app: {app.url}")
