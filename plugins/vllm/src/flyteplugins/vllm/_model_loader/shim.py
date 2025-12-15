import logging
from typing import Generator

import torch
import vllm
import vllm.entrypoints.cli.main
from flyte.app.extras._model_loader.config import (
    LOCAL_MODEL_PATH,
    REMOTE_MODEL_PATH,
    STREAM_SAFETENSORS,
)
from flyte.app.extras._model_loader.loader import SafeTensorsStreamer, prefetch
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

logger = logging.getLogger(__name__)


@register_model_loader("flyte-vllm-streaming")
class FlyteModelLoader(DefaultModelLoader):
    """Custom model loader for streaming model weights from object storage."""

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # Try to load weights using the Flyte SafeTensorsLoader. Fallback to the default loader otherwise.
        try:
            streamer = SafeTensorsStreamer(REMOTE_MODEL_PATH, LOCAL_MODEL_PATH)
        except ValueError:
            yield from super()._get_weights_iterator(source)
        else:
            for name, tensor in streamer.get_tensors():
                yield source.prefix + name, tensor

    def download_model(self, model_config: ModelConfig) -> None:
        # This model loader supports streaming only
        pass

    def _load_sharded_model(self, vllm_config: VllmConfig, model_config: ModelConfig) -> torch.nn.Module:
        # Forked from: https://github.com/vllm-project/vllm/blob/99d01a5e3d5278284bad359ac8b87ee7a551afda/vllm/model_executor/model_loader/loader.py#L613
        # Sanity checks
        tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        rank = get_tensor_model_parallel_rank()
        if rank >= tensor_parallel_size:
            raise ValueError(f"Invalid rank {rank} for tensor parallel size {tensor_parallel_size}")
        with set_default_torch_dtype(vllm_config.model_config.dtype):  # type: ignore[arg-type]
            with torch.device(vllm_config.device_config.device):  # type: ignore[arg-type]
                model_loader = DummyModelLoader(load_config=vllm_config.load_config)
                model = model_loader.load_model(vllm_config=vllm_config, model_config=model_config)
                for i, (name, module) in enumerate(model.named_modules()):
                    print(i, name, module)
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
            state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
            streamer = SafeTensorsStreamer(
                REMOTE_MODEL_PATH,
                LOCAL_MODEL_PATH,
                rank=rank,
                tensor_parallel_size=tensor_parallel_size,
            )
            for name, tensor in streamer.get_tensors():
                # If loading with LoRA enabled, additional padding may
                # be added to certain parameters. We only load into a
                # narrowed view of the parameter data.
                param_data = state_dict[name].data
                param_shape = state_dict[name].shape
                for dim, size in enumerate(tensor.shape):
                    if size < param_shape[dim]:
                        param_data = param_data.narrow(dim, 0, size)
                if tensor.shape != param_shape:
                    logger.warning(
                        "loading tensor of shape %s into parameter '%s' of shape %s",
                        tensor.shape,
                        name,
                        param_shape,
                    )
                param_data.copy_(tensor)
                state_dict.pop(name)
            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")
        return model.eval()

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
    ) -> torch.nn.Module:
        logger.info("Loading model with FlyteModelLoader")
        if vllm_config.parallel_config.tensor_parallel_size > 1:
            return self._load_sharded_model(vllm_config, model_config)
        else:
            return super().load_model(vllm_config, model_config)


async def _get_model_files():
    import flyte.storage as storage

    if not await storage.exists(REMOTE_MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {REMOTE_MODEL_PATH}")

    await prefetch(
        REMOTE_MODEL_PATH,
        LOCAL_MODEL_PATH,
        exclude_safetensors=STREAM_SAFETENSORS,
    )


def main():
    import asyncio

    # TODO: add CLI here to be able to pass in serialized inputs from AppEnvironment
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Prefetch the model
    asyncio.run(_get_model_files())

    vllm.entrypoints.cli.main.main()
