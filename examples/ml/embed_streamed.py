"""Stream a transformers model's weights from object storage straight onto the GPU.

The lower-level counterpart of ``batch_inference_streamed.py``, for when the model
runs in plain PyTorch / ``transformers`` instead of vLLM (embeddings, classifiers, custom
heads). ``load_hf_model`` builds the model with its parameters on the ``meta`` device,
so nothing is allocated yet, then streams each safetensors tensor from the bucket and
copies it onto the GPU as soon as it arrives. The weights never touch local disk and are
never held whole in host memory.

``ModelStreamer`` is the piece underneath, for models that are not ``transformers``
models::

    from flyte.extras.model_streamer import ModelStreamer, empty_weights

    with empty_weights():
        model = MyModel()
    await ModelStreamer("s3://bucket/my-model").load_into(model, device="cuda")

Run (stages the model into object storage once, then embeds on reusable GPU workers):
    flyte run embed_streamed.py main
"""

import asyncio

from async_lru import alru_cache

import flyte
from flyte.io import Dir

MODEL = "BAAI/bge-base-en-v1.5"

image = flyte.Image.from_debian_base(name="model-streamer-transformers").with_pip_packages(
    "torch",
    "transformers",
    "huggingface_hub",
    "async-lru",
    "unionai-reuse",
)

stage_env = flyte.TaskEnvironment(
    name="stage_embedder",
    resources=flyte.Resources(cpu=2, memory="4Gi", disk="16Gi"),
    image=image,
    cache="auto",
)

gpu_env = flyte.TaskEnvironment(
    name="streamed_embedder",
    resources=flyte.Resources(cpu=4, memory="8Gi", gpu="L4:1"),
    image=image,
    reusable=flyte.ReusePolicy(replicas=2, concurrency=8),
)

driver_env = flyte.TaskEnvironment(
    name="embed_driver",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=image,
    depends_on=[stage_env, gpu_env],
)


@stage_env.task
async def stage_model(repo_id: str) -> Dir:
    """Copy a Hub model into the object store once; cached on ``repo_id``."""
    from huggingface_hub import snapshot_download

    local = await asyncio.to_thread(snapshot_download, repo_id, allow_patterns=["*.safetensors", "*.json", "*.txt"])
    return await Dir.from_local(local)


@alru_cache(maxsize=1)
async def get_model(model_path: str):
    """Load once per container: config + tokenizer to disk, weights straight to the GPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from flyte.extras.model_streamer import load_hf_model

    model, local_dir = await load_hf_model(model_path, model_class=AutoModel, device="cuda", dtype=torch.float16)
    return model, AutoTokenizer.from_pretrained(local_dir)


@gpu_env.task
async def embed(model_path: str, texts: list[str]) -> list[list[float]]:
    import torch

    model, tokenizer = await get_model(model_path)
    batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        cls = model(**batch).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(cls, dim=-1).float().cpu().tolist()


@driver_env.task
async def main(model: str = MODEL, num_texts: int = 1_000, chunk_size: int = 100) -> int:
    model_dir = await stage_model(model)
    texts = [f"Document {i}: object storage, streamed to the GPU." for i in range(num_texts)]
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    vectors = await asyncio.gather(*(embed(model_dir.path, chunk) for chunk in chunks))
    return sum(len(v) for v in vectors)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
