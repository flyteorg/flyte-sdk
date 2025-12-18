from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte.app import Input, RunOutput
from flyte.app._types import Port
from flyte.models import SerializationContext

DEFAULT_SGLANG_IMAGE = (
    flyte.Image.from_debian_base(name="sglang-app-image", python_version=(3, 12))
    # install system dependencies, including CUDA toolkit
    .with_apt_packages("libnuma-dev", "wget")
    .with_commands(
        [
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
            "apt-get install -y cuda-toolkit-12-8",
        ]
    )
    # install flash-infer
    .with_pip_packages("flashinfer-python", "flashinfer-cubin")
    .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu128")
    .with_pip_packages("flyteplugins-sglang", pre=True)
    .with_env_vars({"CUDA_HOME": "/usr/local/cuda-12.8"})
)


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class SGLangAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by SGLang for serving large language models.

    This environment sets up an SGLang server with the specified model and configuration.

    :param name: The name of the application.
    :param container_image: The container image to use for the application.
    :param port: Port application listens to. Defaults to 8000 for SGLang.
    :param requests: Compute resource requests for application.
    :param secrets: Secrets that are requested for application.
    :param limits: Compute resource limits for application.
    :param env_vars: Environment variables to set for the application.
    :param scaling: Scaling configuration for the app environment.
    :param domain: Domain to use for the app.
    :param cluster_pool: The target cluster_pool where the app should be deployed.
    :param requires_auth: Whether the public URL requires authentication.
    :param type: Type of app.
    :param extra_args: Extra args to pass to `python -m sglang.launch_server`. See
        https://docs.sglang.io/advanced_features/server_arguments.html for details.
    :param model_path: Remote path to model (e.g., s3://bucket/path/to/model).
    :param model_hf_path: Hugging Face path to model (e.g., Qwen/Qwen3-0.6B).
    :param model_id: Model id that is exposed by SGLang.
    :param stream_model: Set to True to stream model from blob store to the GPU directly.
        If False, the model will be downloaded to the local file system first and then loaded
        into the GPU.
    """

    port: int | Port = 8080
    type: str = "SGLang"
    extra_args: str | list[str] = ""
    model_path: str | RunOutput = ""
    model_hf_path: str = ""
    model_id: str = ""
    stream_model: bool = True
    image: str | Image | Literal["auto"] = DEFAULT_SGLANG_IMAGE
    _model_mount_path: str = field(default="/root/flyte", init=False)

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

        if self.model_id == "":
            raise ValueError("model_id must be defined")

        if self.model_path == "" and self.model_hf_path == "":
            raise ValueError("model_path or model_hf_path must be defined")
        if self.model_path != "" and self.model_hf_path != "":
            raise ValueError("model_path and model_hf_path cannot be set at the same time")

        if self.model_hf_path:
            self._model_mount_path = self.model_hf_path

        if self.args:
            raise ValueError("args cannot be set for SGLangAppEnvironment. Use `extra_args` to add extra arguments.")

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = self.extra_args

        self.args = [
            "sglang-fserve",
            "--model-path",
            self._model_mount_path,
            "--served-model-name",
            self.model_id,
            "--port",
            str(self.get_port().port),
            *extra_args,
        ]

        if self.inputs:
            raise ValueError("inputs cannot be set for SGLangAppEnvironment")

        input_kwargs = {}
        if self.stream_model:
            self.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] = "true"
            input_kwargs["env_var"] = "FLYTE_MODEL_LOADER_REMOTE_MODEL_PATH"
            input_kwargs["download"] = False
        else:
            self.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] = "false"
            input_kwargs["download"] = True
            input_kwargs["mount"] = self._model_mount_path

        if self.model_path:
            self.inputs = [Input(name="model_path", value=self.model_path, **input_kwargs)]

        self.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] = self._model_mount_path
        self.links = [flyte.app.Link(path="/docs", title="SGLang OpenAPI Docs", is_relative=True)]

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_SGLANG_IMAGE

        super().__post_init__()

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        """Return the container arguments for SGLang."""
        if isinstance(self.args, str):
            return shlex.split(self.args)
        return self.args or []

    def clone_with(
        self,
        name: str,
        image: Optional[Union[str, Image, Literal["auto"]]] = None,
        resources: Optional[Resources] = None,
        env_vars: Optional[dict[str, str]] = None,
        secrets: Optional[SecretRequest] = None,
        depends_on: Optional[list[Environment]] = None,
        description: Optional[str] = None,
        interruptible: Optional[bool] = None,
        **kwargs: Any,
    ) -> SGLangAppEnvironment:
        port = kwargs.pop("port", None)
        extra_args = kwargs.pop("extra_args", None)
        if "model_path" in kwargs:
            set_model_path = True
            model_path = kwargs.pop("model_path", "") or ""
        else:
            set_model_path = False
            model_path = self.model_path
        if "model_hf_path" in kwargs:
            set_model_hf_path = True
            model_hf_path = kwargs.pop("model_hf_path", "") or ""
        else:
            set_model_hf_path = False
            model_hf_path = self.model_hf_path
        model_id = kwargs.pop("model_id", None)
        stream_model = kwargs.pop("stream_model", None)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        kwargs = self._get_kwargs()
        kwargs["name"] = name
        kwargs["args"] = None
        kwargs["inputs"] = None
        if image is not None:
            kwargs["image"] = image
        if resources is not None:
            kwargs["resources"] = resources
        if env_vars is not None:
            kwargs["env_vars"] = env_vars
        if secrets is not None:
            kwargs["secrets"] = secrets
        if depends_on is not None:
            kwargs["depends_on"] = depends_on
        if description is not None:
            kwargs["description"] = description
        if interruptible is not None:
            kwargs["interruptible"] = interruptible
        if port is not None:
            kwargs["port"] = port
        if extra_args is not None:
            kwargs["extra_args"] = extra_args
        if set_model_path:
            kwargs["model_path"] = model_path
        if set_model_hf_path:
            kwargs["model_hf_path"] = model_hf_path
        if model_id is not None:
            kwargs["model_id"] = model_id
        if stream_model is not None:
            kwargs["stream_model"] = stream_model
        return replace(self, **kwargs)
