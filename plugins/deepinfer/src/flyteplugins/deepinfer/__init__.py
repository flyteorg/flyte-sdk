"""DeepInfer plugin for Flyte.

Run [DeepSeek's DeepInfer](https://github.com/deepseek-ai/DeepInfer) offline (batch)
inference in Flyte tasks. DeepInfer is the inference framework behind DeepSeek's
production serving (optimised kernels, MoE expert-parallel load balancing, speculative
decoding, structured output); it supports DeepSeek-V4.1-Flash on NVIDIA SM100 GPUs
(B200 / GB200) and Huawei Ascend 950 NPUs.

```python
import flyte
from flyteplugins.deepinfer import DeepInfer, deepinfer_image

env = flyte.TaskEnvironment(
    name="deepinfer",
    resources=flyte.Resources(gpu="B200:8"),
    image=deepinfer_image(),
)
llm = DeepInfer("deepseek-ai/DeepSeek-V4.1-Flash-1017")

@env.task
async def complete(prompts: list[str]) -> list[str]:
    return await llm.generate.aio(prompts, temperature=0, max_tokens=512)
```

- `DeepInfer`: a lazily started pool of engines (one per GPU by default) that stays
  warm for the life of the task process; `generate` returns results in input order.
- `deepinfer_image`: a container image that builds DeepInfer from source at a pinned
  commit, or layers the plugin onto an image that already has it.
- `clustered_deepinfer_environment`: a multi-node `ClusteredTaskEnvironment` shaped to
  DeepInfer's one-launcher-per-node model.

DeepInfer is built from source, not published to PyPI, and targets specific hardware,
so nothing here imports DeepInfer at module load.
"""

from flyteplugins.deepinfer._clustered import clustered_deepinfer_environment
from flyteplugins.deepinfer._engine import DeepInfer
from flyteplugins.deepinfer._image import deepinfer_image

__all__ = [
    "DeepInfer",
    "clustered_deepinfer_environment",
    "deepinfer_image",
]
