"""Stream safetensors model weights from object storage straight onto a device.

Weights are fetched as parallel byte ranges and handed over tensor by tensor
as each one completes. They never touch local disk, and host memory holds only
the tensors in flight, so a model's load time is close to its download time.

- `ModelStreamer`: the streamer itself (async and sync iterators, a
  state dict, or straight into an `nn.Module`).
- `load_hf_model`: a `transformers` model built on `meta` and filled
  on the GPU.
- `flyteplugins.vllm.model_streamer` (in `flyteplugins-vllm`): a vLLM
  `load_format` for `vllm.LLM` / `AsyncLLMEngine`.

Requires `torch`; `load_hf_model` also needs `transformers`, and the
vLLM integration needs `vllm`.
"""

from flyte.extras.model_streamer._hf import load_hf_model
from flyte.extras.model_streamer._streamer import (
    LoadResult,
    ModelStreamer,
    empty_weights,
    missing_parameters,
)

__all__ = ["LoadResult", "ModelStreamer", "empty_weights", "load_hf_model", "missing_parameters"]
