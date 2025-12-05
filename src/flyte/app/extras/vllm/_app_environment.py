import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Union

import rich.repr

import flyte.app
from flyte.app._input import Input
from flyte.models import SerializationContext

DEFAULT_VLLM_IMAGE = "ghcr.io/unionai/serving-vllm:py3.12-latest"


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class VLLMAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by vLLM for serving large language models.

    This environment sets up a vLLM server with the specified model and configuration.

    :param name: The name of the application.
    :param container_image: The container image to use for the application.
    :param port: Port application listens to. Defaults to 8000 for vLLM.
    :param requests: Compute resource requests for application.
    :param secrets: Secrets that are requested for application.
    :param limits: Compute resource limits for application.
    :param env_vars: Environment variables to set for the application.
    :param scaling: Scaling configuration for the app environment.
    :param domain: Domain to use for the app.
    :param cluster_pool: The target cluster_pool where the app should be deployed.
    :param requires_auth: Whether the public URL requires authentication.
    :param type: Type of app.
    :param extra_args: Extra args to pass to `vllm serve`. See
        https://docs.vllm.ai/en/stable/serving/engine_args.html
        or run `vllm serve --help` for details.
    :param model: Remote path to model (e.g., s3://bucket/path/to/model).
    :param model_id: Model id that is exposed by vllm.
    :param stream_model: Set to True to stream model from blob store to the GPU directly.
        If False, the model will be downloaded to the local file system first and then loaded
        into the GPU.
    """

    port: int = 8000
    type: str = "vLLM"
    extra_args: Union[str, List[str]] = ""
    model: str = ""
    model_id: str = ""
    stream_model: bool = True
    _model_mount_path: str = field(default="/root/flyte", init=False)

    def __post_init__(self):
        if self.model_id == "":
            raise ValueError("model_id must be defined")

        if self.model == "":
            raise ValueError("model must be defined")

        if self.args:
            raise ValueError("args cannot be set for VLLMAppEnvironment. Use `extra_args` to add extra arguments.")

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = self.extra_args

        stream_model_args = []
        if self.stream_model:
            stream_model_args.extend(["--load-format", "flyte-streaming"])

        self.args = [
            "flyte-vllm-model-loader",
            "serve",
            self._model_mount_path,
            "--served-model-name",
            self.model_id,
            "--port",
            str(self.port),
            *stream_model_args,
            *extra_args,
        ]

        if self.inputs:
            raise ValueError("inputs cannot be set for VLLMAppEnvironment")

        input_kwargs = {}
        if self.stream_model:
            self.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] = "true"
            input_kwargs["env_var"] = "FLYTE_MODEL_LOADER_REMOTE_MODEL_PATH"
            input_kwargs["download"] = False
        else:
            self.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] = "false"
            input_kwargs["download"] = True
            input_kwargs["mount"] = self._model_mount_path

        self.inputs = [Input(name="model", value=self.model, **input_kwargs)]
        self.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] = self._model_mount_path

        if self.image is None:
            self.image = DEFAULT_VLLM_IMAGE

        super().__post_init__()

    def container_args(self, serialization_context: SerializationContext) -> List[str]:
        """Return the container arguments for vLLM."""
        if isinstance(self.args, str):
            return shlex.split(self.args)
        return self.args or []

