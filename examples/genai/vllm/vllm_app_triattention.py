"""
A vLLM app example deploying Qwen3-32B with TriAttention KV cache compression.

TriAttention (https://github.com/WeianMao/triattention) uses trigonometric
frequency-domain compression to reduce KV cache memory by up to 10.7x while
matching full attention accuracy. This enables serving large models like
Qwen3-32B (INT4) on a single A10G GPU with long context support.

Deploy
------

Deploy this app using the Flyte CLI:

```
flyte deploy examples/genai/vllm/vllm_app_triattention.py vllm_app_triattention
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
    model="qwen3-32b-int4",
    messages=[
        {"role": "user", "content": "Solve the equation: x^2 + 5x + 6 = 0"}
    ],
)
print(response.choices[0].message.content)
```
"""

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app

# Define the vLLM app environment for Qwen3-32B INT4 with TriAttention
vllm_app = VLLMAppEnvironment(
    name="qwen3-32b-triattention",
    model_hf_path="Qwen/Qwen3-32B-AWQ",
    model_id="qwen3-32b-int4",
    resources=flyte.Resources(cpu="4", memory="24Gi", gpu="A10G:1", disk="50Gi"),
    image=(
        flyte.Image.from_debian_base(
            name="vllm-triattention-image",
            install_flyte=False,
        )
        .with_pip_packages("vllm==0.19.0", "transformers==4.57.6")
        .with_apt_packages("git")
        .with_pip_packages("triattention @ git+https://github.com/WeianMao/triattention.git")
        .with_pip_packages(
            "flash-attn",
            extra_args="--no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3",
        )
        .with_pip_packages("flyteplugins-vllm")
    ),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,
    ),
    requires_auth=False,
    env_vars={
        "ENABLE_TRIATTENTION": "true",
        "TRIATTN_RUNTIME_KV_BUDGET": "2048",
        "TRIATTN_RUNTIME_DIVIDE_LENGTH": "128",
        "TRIATTN_RUNTIME_WINDOW_SIZE": "128",
        "TRIATTN_RUNTIME_SPARSE_STATS_PATH": "triattention/vllm/stats/qwen3_32b_int4_stats.pt",
    },
    extra_args=[
        "--max-model-len",
        "32768",
        "--enforce-eager",
        "--trust-remote-code",
        "--quantization",
        "awq",
        "--gpu-memory-utilization",
        "0.9",
    ],
)


if __name__ == "__main__":
    import flyte.prefetch

    flyte.init_from_config()

    # Prefetch the Qwen3-32B-AWQ model into flyte object store
    run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-32B-AWQ")
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
