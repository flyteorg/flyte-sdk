from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte._logging import logger
from flyte.app import ArtifactValue, Parameter, RunOutput
from flyte.app._types import Port
from flyte.models import SerializationContext

from flyteplugins.vllm._constants import (
    FLASHINFER_JIT_CACHE_INDEX_URL,
    FLASHINFER_VERSION,
    VLLM_MIN_VERSION_STR,
)

DEFAULT_VLLM_IMAGE = (
    flyte.Image.from_debian_base(name="vllm-app-image")
    # install the vllm flyte plugin
    .with_pip_packages("flyteplugins-vllm", pre=True)
    # install vllm in a separate layer due to dependency conflict with flyte (protovalidate)
    .with_pip_packages(f"vllm=={VLLM_MIN_VERSION_STR}")
    # FlashInfer's prebuilt kernels. This image ships the CUDA *runtime* (via the torch/vLLM
    # wheels) but no toolkit, so anything that JIT-compiles a kernel at startup dies with
    # "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist". vLLM's
    # top-k/top-p sampler is exactly such a path -- it builds its sampling module during
    # warmup, after the weights load, so the failure looks like a late crash rather than a
    # missing dependency. Shipping the jit-cache means nothing has to compile at runtime.
    #
    # Deliberately after the vLLM layer and pinned to vLLM's own flashinfer version: layers
    # resolve independently, so an earlier unpinned layer would just be overwritten by vLLM's
    # exact pin and leave the cache mismatched. ``flashinfer-cubin`` is absent because it
    # publishes no release matching FLASHINFER_VERSION.
    .with_pip_packages(
        f"flashinfer-jit-cache=={FLASHINFER_VERSION}",
        index_url=FLASHINFER_JIT_CACHE_INDEX_URL,
    )
)


def _shell_safe(args: list[str]) -> list[str]:
    """Quote args that have to survive a trip through a shell.

    ``fserve`` runs the app with ``Popen(" ".join(args), shell=True)`` (flyte/_bin/serve.py),
    so any token carrying spaces or quotes -- vLLM's ``--speculative-config`` JSON blob being
    the obvious one -- reaches the engine mangled unless it is quoted here. ``shlex.quote`` is
    the identity function for ordinary tokens, so this is a no-op for everything else.

    Tokens starting with ``$`` are left alone: ``fserve`` expands those against the container
    environment *before* joining, and quoting would turn the marker into a literal.
    """
    return [arg if arg.startswith("$") else shlex.quote(arg) for arg in args]


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class VLLMAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by vLLM for serving large language models.

    This environment sets up a vLLM server with the specified model and configuration.

    Args:
        name: The name of the application.
        container_image: The container image to use for the application.
        port: Port application listens to. Defaults to 8000 for vLLM.
        requests: Compute resource requests for application.
        secrets: Secrets that are requested for application.
        limits: Compute resource limits for application.
        env_vars: Environment variables to set for the application.
        scaling: Scaling configuration for the app environment.
        domain: Domain to use for the app.
        cluster_pool: The target cluster_pool where the app should be deployed.
        requires_auth: Whether the public URL requires authentication.
        type: Type of app.
        extra_args: Extra args to pass to `vllm serve`. See
            https://docs.vllm.ai/en/stable/configuration/engine_args
            or run `vllm serve --help` for details.
        model_path: Remote path to model (e.g., s3://bucket/path/to/model), or a
            `RunOutput`/`ArtifactValue` resolved at deploy time.
        model_hf_path: Hugging Face path to model (e.g., Qwen/Qwen3-0.6B).
        model_id: Model id that is exposed by vllm.
        stream_model: When `model_path` is set, use True to stream weights from object
            storage to the GPU (Flyte custom loader). Ignored for `model_hf_path`-only apps,
            which always use vLLM's normal Hugging Face download path. If False with `model_path`,
            the model is downloaded to the local filesystem first, then loaded. Also ignored when
            a draft model is configured -- see `draft_model_path`.
        draft_model_path: Remote path to the draft model (speculator) used for speculative
            decoding, or a `RunOutput`/`ArtifactValue` resolved at deploy time. The weights are
            downloaded alongside the target model and passed to vLLM as the `model` key of
            `--speculative-config`. Requires `speculative_config`.
        draft_model_hf_path: Hugging Face path to the draft model, as an alternative to
            `draft_model_path` (e.g., Qwen/Qwen3-0.6B).
        speculative_config: vLLM speculative decoding configuration, serialized into the
            `--speculative-config` JSON blob. The `model` key is filled in from
            `draft_model_path`/`draft_model_hf_path` and must not be set here. Draft-model-free
            methods (e.g. `{"method": "ngram", "num_speculative_tokens": 5}`) need no draft model.
            See https://docs.vllm.ai/en/stable/features/spec_decode/.
    """

    port: int | Port = 8080
    type: str = "vLLM"
    extra_args: str | list[str] = ""
    model_path: str | RunOutput | ArtifactValue = ""
    model_hf_path: str = ""
    model_id: str = ""
    stream_model: bool = True
    draft_model_path: str | RunOutput | ArtifactValue = ""
    draft_model_hf_path: str = ""
    speculative_config: Optional[dict[str, Any]] = None
    image: str | Image | Literal["auto"] = DEFAULT_VLLM_IMAGE
    # Under /tmp, and that is not cosmetic: ``fserve`` materializes each mounted Parameter
    # through ``_ensure_dest_writable``, which needs the *image's* user to be able to create the
    # parent directory. The released Flyte base image runs non-root, so a mount at the
    # filesystem root -- or under /root -- fails with "Permission denied" before the engine ever
    # starts. /tmp is writable for any user and lives on the same overlay filesystem the weights
    # are already budgeted against by ``disk=``.
    _model_mount_path: str = field(default="/tmp/flyte/model", init=False)
    _draft_model_mount_path: str = field(default="/tmp/flyte/draft-model", init=False)

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

        if self._server is not None:
            raise ValueError("server function cannot be set for VLLMAppEnvironment")

        if self._on_startup is not None:
            raise ValueError("on_startup function cannot be set for VLLMAppEnvironment")

        if self._on_shutdown is not None:
            raise ValueError("on_shutdown function cannot be set for VLLMAppEnvironment")

        if self.model_id == "":
            raise ValueError("model_id must be defined")

        if self.model_path == "" and self.model_hf_path == "":
            raise ValueError("model_path or model_hf_path must be defined")
        if self.model_path != "" and self.model_hf_path != "":
            raise ValueError("model_path and model_hf_path cannot be set at the same time")

        if self.draft_model_path != "" and self.draft_model_hf_path != "":
            raise ValueError("draft_model_path and draft_model_hf_path cannot be set at the same time")

        if self.model_hf_path:
            self._model_mount_path = self.model_hf_path

        if self.args:
            raise ValueError("args cannot be set for VLLMAppEnvironment. Use `extra_args` to add extra arguments.")

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = list(self.extra_args)

        speculative_args = self._speculative_args()

        # Flyte streaming requires a remote ``model_path`` (and the model-loader env). HF-only
        # apps must use vLLM's default loaders regardless of ``stream_model``.
        #
        # A draft model rules streaming out entirely: ``--load-format flyte-vllm-streaming`` is
        # process-wide and ``FLYTE_MODEL_LOADER_REMOTE_MODEL_PATH``/``..._LOCAL_MODEL_PATH`` are a
        # single global pair, so the loader can only ever describe one set of weights. With two
        # models in one process it would feed the target's tensors to the draft model.
        use_flyte_blob_streaming = bool(self.stream_model and self.model_path and not self._has_draft_model)
        if self.stream_model and self.model_path and self._has_draft_model:
            logger.warning(
                "stream_model is not supported alongside a draft model: the Flyte streaming loader is "
                "process-wide and describes a single set of weights. Falling back to downloading both "
                "the target and the draft model to the container filesystem."
            )

        stream_model_args: list[str] = []
        if use_flyte_blob_streaming:
            stream_model_args.extend(["--load-format", "flyte-vllm-streaming"])

        self.args = _shell_safe(
            [
                "vllm-fserve",
                "serve",
                self._model_mount_path,
                "--served-model-name",
                self.model_id,
                "--port",
                str(self.get_port().port),
                *stream_model_args,
                *speculative_args,
                *extra_args,
            ]
        )

        if self.parameters:
            raise ValueError("parameters cannot be set for VLLMAppEnvironment")

        input_kwargs: dict[str, Any] = {}
        if use_flyte_blob_streaming:
            self.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] = "true"
            input_kwargs["env_var"] = "FLYTE_MODEL_LOADER_REMOTE_MODEL_PATH"
            input_kwargs["download"] = False
        else:
            self.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] = "false"
            input_kwargs["download"] = True
            input_kwargs["mount"] = self._model_mount_path

        parameters: list[Parameter] = []
        if self.model_path:
            parameters.append(Parameter(name="model_path", value=self.model_path, **input_kwargs))
        if self.draft_model_path:
            parameters.append(
                Parameter(
                    name="draft_model_path",
                    value=self.draft_model_path,
                    download=True,
                    mount=self._draft_model_mount_path,
                )
            )
        if parameters:
            self.parameters = parameters

        self.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] = self._model_mount_path
        self.links = [flyte.app.Link(path="/docs", title="vLLM OpenAPI Docs", is_relative=True), *self.links]

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_VLLM_IMAGE

        super().__post_init__()

    @property
    def _has_draft_model(self) -> bool:
        return bool(self.draft_model_path or self.draft_model_hf_path)

    def _speculative_args(self) -> list[str]:
        """Render ``--speculative-config``, pointing it at the draft model when there is one."""
        if self._has_draft_model and self.speculative_config is None:
            raise ValueError(
                "speculative_config must be defined when a draft model is set. It selects the "
                'speculative decoding method and its knobs, e.g. {"method": "eagle3", '
                '"num_speculative_tokens": 3}.'
            )

        if self.speculative_config is None:
            return []

        config = dict(self.speculative_config)
        if self._has_draft_model:
            if "model" in config:
                raise ValueError(
                    "speculative_config cannot set 'model'. Use draft_model_path or "
                    "draft_model_hf_path instead, so that the weights are mounted into the container."
                )
            config["model"] = self.draft_model_hf_path or self._draft_model_mount_path

        return ["--speculative-config", json.dumps(config)]

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        """Return the container arguments for vLLM."""
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
    ) -> VLLMAppEnvironment:
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
        if "draft_model_path" in kwargs:
            set_draft_model_path = True
            draft_model_path = kwargs.pop("draft_model_path", "") or ""
        else:
            set_draft_model_path = False
            draft_model_path = self.draft_model_path
        if "draft_model_hf_path" in kwargs:
            set_draft_model_hf_path = True
            draft_model_hf_path = kwargs.pop("draft_model_hf_path", "") or ""
        else:
            set_draft_model_hf_path = False
            draft_model_hf_path = self.draft_model_hf_path
        set_speculative_config = "speculative_config" in kwargs
        speculative_config = kwargs.pop("speculative_config", None)
        model_id = kwargs.pop("model_id", None)
        stream_model = kwargs.pop("stream_model", None)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        kwargs = self._get_kwargs()
        kwargs["name"] = name
        kwargs["args"] = None
        kwargs["parameters"] = None
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
        if set_draft_model_path:
            kwargs["draft_model_path"] = draft_model_path
        if set_draft_model_hf_path:
            kwargs["draft_model_hf_path"] = draft_model_hf_path
        if set_speculative_config:
            kwargs["speculative_config"] = speculative_config
        if model_id is not None:
            kwargs["model_id"] = model_id
        if stream_model is not None:
            kwargs["stream_model"] = stream_model
        return replace(self, **kwargs)
