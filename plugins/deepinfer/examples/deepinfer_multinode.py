"""Multi-node DeepInfer: one expert-parallel pool spanning several nodes.

Use this when you want DeepInfer's MoE expert-parallel pool to span more GPUs than one
node has (for example, to free per-GPU memory for longer contexts or bigger batches).
`clustered_deepinfer_environment` runs one DeepInfer node-launcher per pod; inside it,
`run_engines` spawns this node's engines, and `create_llm` defaults `ep_size` to every
GPU across all nodes.

Two Flyte rules shape the code:

- Only rank 0's return value becomes a clustered task's output, and `run_engines`
  returns just this node's ranks. So every node writes its own results to a shared
  directory, and a regular driver task merges them.
- A clustered task cannot launch other tasks, so the driver is a plain
  `TaskEnvironment` that `depends_on` the clustered one.

This example also starts from a **prebuilt** DeepInfer image rather than compiling it:
set `DEEPINFER_IMAGE` below to your registry's image.

    flyte run examples/deepinfer_multinode.py multinode_generate --prompts '["Hello!"]'
"""

import functools
import json
import os

import flyte
import flyte.io

from flyteplugins.deepinfer import (
    clustered_deepinfer_environment,
    create_llm,
    get_actual_num_gpus,
    run_engines,
    sampling_params,
    setup_common_env,
)

# --- Knobs ---------------------------------------------------------------------------
# A team-built image that already contains a compiled `deepinfer`.
DEEPINFER_IMAGE = "ghcr.io/your-org/deepinfer:sm100"
NODES = 2  # pods == DeepInfer nodes
GPUS_PER_NODE = 8

# The plugin is layered on top of the prebuilt image (a `from_base` image must be
# cloned as extendable before adding layers).
image = (
    flyte.Image.from_base(DEEPINFER_IMAGE)
    .clone(name="deepinfer-multinode", extendable=True)
    .with_pip_packages("flyteplugins-deepinfer", pre=True)
)

cluster = clustered_deepinfer_environment(
    nodes=NODES,
    image=image,
    resources=flyte.Resources(gpu=GPUS_PER_NODE),  # per node: one process drives every GPU
)

driver = flyte.TaskEnvironment(
    name="deepinfer-multinode-driver",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=image,
    depends_on=[cluster],
)


def _generate_shard(model: str, prompts: list[str], params: dict, dp_size: int, dp_rank: int, out) -> None:
    from deepinfer.sampling_params import SamplingParams

    llm = create_llm(model, dp_rank)  # ep_size = WORLD_SIZE x GPUs per node
    indices = list(range(dp_rank, len(prompts), dp_size))
    if indices:
        outputs = llm.generate(
            prompts=[prompts[i] for i in indices],
            sampling_params=[SamplingParams(**params) for _ in indices],
            use_tqdm=False,
        )
        tokenizer = llm.get_tokenizer()
        out.extend(
            (i, tokenizer.decode([tok.token_id for tok in output], skip_special_tokens=True))
            for i, output in zip(indices, outputs)
        )
    llm.close()


@cluster.task
async def generate_on_cluster(prompts: list[str], model: str, max_tokens: int, out_dir: str) -> None:
    """Runs once per node; writes this node's `(index, text)` results to `out_dir`."""
    node_rank = int(os.environ["RANK"])
    dp_size = int(os.environ["WORLD_SIZE"]) * get_actual_num_gpus()
    results = run_engines(
        functools.partial(_generate_shard, model, prompts, sampling_params(temperature=0, max_tokens=max_tokens)),
        dp_size=dp_size,
        initializer=setup_common_env,
    )
    shard = flyte.io.File.from_existing_remote(f"{out_dir}/node-{node_rank}.json")
    async with shard.open("wb") as fh:
        await fh.write(json.dumps(results).encode("utf-8"))


@driver.task
async def multinode_generate(
    prompts: list[str],
    model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    max_tokens: int = 512,
) -> list[str]:
    out_dir = flyte.io.Dir.new_remote("deepinfer-results")
    await generate_on_cluster(prompts, model, max_tokens, out_dir.path)

    texts: dict[int, str] = {}
    async for shard in flyte.io.Dir.from_existing_remote(out_dir.path).walk():
        async with shard.open("rb") as fh:
            texts.update((i, text) for i, text in json.loads(bytes(await fh.read()).decode("utf-8")))
    return [texts[i] for i in range(len(prompts))]


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(multinode_generate, prompts=["Hello!", "Explain expert parallelism briefly."])
    print("Run URL:", run.url)
