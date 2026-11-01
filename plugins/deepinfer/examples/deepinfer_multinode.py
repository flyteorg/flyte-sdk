"""Multi-node DeepInfer: one expert-parallel pool spanning several nodes.

Use this when DeepInfer's MoE expert-parallel pool should span more GPUs than one node
has (for example, to free per-GPU memory for longer contexts or bigger batches).
`clustered_deepinfer_environment` runs the task body once per node; there, `DeepInfer`
starts that node's engines, with `ep_size` defaulting to every GPU across all nodes.

Two Flyte rules shape the code:

- Only rank 0's return value becomes a clustered task's output, and each node only
  produces its own engines' results. So every node writes its results to a shared
  directory, and a regular driver task merges them.
- Only a regular task can launch a clustered one, so the driver is a plain
  `TaskEnvironment` that `depends_on` the clustered one.

This example starts from a **prebuilt** DeepInfer image rather than compiling it.

    flyte run examples/deepinfer_multinode.py multinode_generate --prompts '["Hello!"]'
"""

import json
import os

import flyte
import flyte.io

from flyteplugins.deepinfer import DeepInfer, clustered_deepinfer_environment, deepinfer_image

# --- Knobs ----------------------------------------------------------------------------
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
GPU = "B200:8"  # per node
NODES = 2
PREBUILT_IMAGE = "ghcr.io/your-org/deepinfer:b200"  # already contains a built `deepinfer`

image = deepinfer_image(base=PREBUILT_IMAGE)

cluster = clustered_deepinfer_environment(nodes=NODES, image=image, resources=flyte.Resources(gpu=GPU))

driver = flyte.TaskEnvironment(
    name="deepinfer-multinode-driver",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=image,
    depends_on=[cluster],
)

llm = DeepInfer(MODEL)


@cluster.task
async def generate_on_cluster(prompts: list[str], max_tokens: int, out_dir: str) -> None:
    """Runs once per node; writes this node's `(index, completion)` pairs to `out_dir`."""
    results = await llm.generate.aio(prompts, temperature=0, max_tokens=max_tokens)
    mine = [(i, text) for i, text in enumerate(results) if text is not None]
    shard = flyte.io.File.from_existing_remote(f"{out_dir}/node-{os.environ['RANK']}.json")
    async with shard.open("wb") as fh:
        await fh.write(json.dumps(mine).encode("utf-8"))


@driver.task
async def multinode_generate(prompts: list[str], max_tokens: int = 512) -> list[str]:
    out_dir = flyte.io.Dir.new_remote("deepinfer-results")
    await generate_on_cluster(prompts, max_tokens, out_dir.path)

    completions: dict[int, str] = {}
    async for shard in flyte.io.Dir.from_existing_remote(out_dir.path).walk():
        async with shard.open("rb") as fh:
            completions.update((i, text) for i, text in json.loads(bytes(await fh.read()).decode("utf-8")))
    return [completions[i] for i in range(len(prompts))]


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(multinode_generate, prompts=["Hello!", "Explain expert parallelism briefly."])
    print("Run URL:", run.url)
