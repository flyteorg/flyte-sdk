import logging
import os
import sys
from typing import Generator

import sglang.srt.model_loader.loader
import torch
from flyte.app.extras._model_loader.config import (
    LOCAL_MODEL_PATH,
    REMOTE_MODEL_PATH,
    STREAM_SAFETENSORS,
)
from flyte.app.extras._model_loader.loader import SafeTensorsStreamer, prefetch
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

try:
    from sglang.srt.server import launch_server
except (ModuleNotFoundError, AttributeError, ImportError):
    try:
        from sglang.launch_server import launch_server
    except (ModuleNotFoundError, AttributeError, ImportError):
        from sglang.launch_server import run_server as launch_server


logger = logging.getLogger(__name__)

_OrigDefaultModelLoader = sglang.srt.model_loader.loader.DefaultModelLoader


class FlyteModelLoader(_OrigDefaultModelLoader):
    """Custom model loader for streaming model weights from object storage."""

    def _get_weights_iterator(self, source) -> Generator[tuple[str, torch.Tensor], None, None]:
        # Try to load weights using the Flyte SafeTensorsStreamer. Fallback to the default loader otherwise.
        try:
            streamer = SafeTensorsStreamer(REMOTE_MODEL_PATH, LOCAL_MODEL_PATH)
        except ValueError:
            return super()._get_weights_iterator(source)
        else:
            for name, tensor in streamer.get_tensors():
                yield source.prefix + name, tensor

    def download_model(self, model_config: ModelConfig) -> None:
        # This model loader supports streaming only
        pass

    def _load_sharded_model(self, *, model_config: ModelConfig, device_config: DeviceConfig) -> torch.nn.Module:
        # Forked from: https://github.com/sgl-project/sglang/blob/1c4e0d2445311f2e635e9dab5a660d982731ad20/python/sglang/srt/model_loader/loader.py#L564
        from sglang.srt.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            model_parallel_is_initialized,
        )
        from sglang.srt.model_loader.loader import (
            ShardedStateLoader,
            _initialize_model,
        )
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        # Sanity checks
        if model_parallel_is_initialized():
            tensor_parallel_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
        else:
            tensor_parallel_size = 1
            rank = 0
        if rank >= tensor_parallel_size:
            raise ValueError(f"Invalid rank {rank} for tensor parallel size {tensor_parallel_size}")
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config)
                for _, module in model.named_modules():
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

    def load_model(self, *, model_config: ModelConfig, device_config: DeviceConfig) -> torch.nn.Module:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        logger.info("Loading model with FlyteModelLoader")
        if get_tensor_model_parallel_world_size() > 1:
            return self._load_sharded_model(model_config=model_config, device_config=device_config)
        else:
            return super().load_model(model_config=model_config, device_config=device_config)


# Monkeypatch the default model loader when streaming is enabled
if REMOTE_MODEL_PATH and STREAM_SAFETENSORS:
    sglang.srt.model_loader.loader.DefaultModelLoader = FlyteModelLoader


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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(f"REMOTE_MODEL_PATH: {REMOTE_MODEL_PATH}")
    logger.info(f"LOCAL_MODEL_PATH: {LOCAL_MODEL_PATH}")
    logger.info(f"STREAM_SAFETENSORS: {STREAM_SAFETENSORS}")

    # Prefetch the model
    if REMOTE_MODEL_PATH:
        logger.info("Prefetching model files...")
        asyncio.run(_get_model_files())

    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
