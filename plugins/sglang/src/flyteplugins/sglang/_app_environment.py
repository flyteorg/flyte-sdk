from __future__ import annotations

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

from flyteplugins.sglang._constants import (
    CUDA_HOME,
    CUDA_TOOLKIT_PACKAGE,
    SGLANG_MIN_VERSION_STR,
    SGLANG_ROUTER_VERSION,
)

DEFAULT_SGLANG_IMAGE = (
    flyte.Image.from_debian_base(name="sglang-app-image")
    # install system dependencies, including CUDA toolkit, which is needed by sglang for compiling the model
    # and rust and cargo for installing sglang
    .with_apt_packages("libnuma-dev", "wget", "curl", "openssl", "pkg-config", "libssl-dev", "build-essential")
    .with_commands(
        [
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
            f"apt-get install -y {CUDA_TOOLKIT_PACKAGE}",
        ]
    )
    .with_commands(["curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . $HOME/.cargo/env"])
    .with_env_vars({"CUDA_HOME": CUDA_HOME, "PATH": f"/root/.cargo/bin:{CUDA_HOME}/bin:$PATH"})
    .with_pip_packages("flyteplugins-sglang", pre=True)
    # No hand-pinned flashinfer layer: SGLang pins an exact flashinfer version with an exact
    # CUDA extra (0.5.16 -> flashinfer_python[cu13]), so installing our own first only gives it
    # something to fight, and a jit-cache built for a different CUDA major is actively wrong.
    # The toolkit installed above covers whatever JIT remains -- a slower first start, but
    # correct, and no index URL to keep in sync by hand.
    #
    # ``pre=True`` is required rather than cosmetic: SGLang depends on flash-attn-4>=4.0.0b18,
    # itself a pre-release, and uv refuses pre-releases by default. Without it the resolver
    # reports the package as simply unsatisfiable.
    .with_pip_packages(f"sglang=={SGLANG_MIN_VERSION_STR}", pre=True)
    # The cache-aware router, used when ``router=True``. Also pre=True: this layer re-resolves
    # against the pre-release deps SGLang just installed and needs the same policy.
    .with_pip_packages(f"sglang-router=={SGLANG_ROUTER_VERSION}", pre=True)
)


def _shell_safe(args: list[str]) -> list[str]:
    """Quote args that have to survive a trip through a shell.

    `fserve` runs the app with `Popen(" ".join(args), shell=True)` (flyte/_bin/serve.py), so
    any token carrying spaces or quotes reaches the engine mangled unless it is quoted here.
    `shlex.quote` is the identity function for ordinary tokens, so this is a no-op for
    everything else.

    Tokens starting with `$` are left alone: `fserve` expands those against the container
    environment *before* joining, and quoting would turn the marker into a literal.
    """
    return [arg if arg.startswith("$") else shlex.quote(arg) for arg in args]


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class SGLangAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by SGLang for serving large language models.

    This environment sets up an SGLang server with the specified model and configuration.

    Args:
        name: The name of the application.
        container_image: The container image to use for the application.
        port: Port application listens to. Defaults to 8000 for SGLang.
        requests: Compute resource requests for application.
        secrets: Secrets that are requested for application.
        limits: Compute resource limits for application.
        env_vars: Environment variables to set for the application.
        scaling: Scaling configuration for the app environment.
        domain: Domain to use for the app.
        cluster_pool: The target cluster_pool where the app should be deployed.
        requires_auth: Whether the public URL requires authentication.
        type: Type of app.
        extra_args: Extra args to pass to `python -m sglang.launch_server`. See
            https://docs.sglang.io/advanced_features/server_arguments.html for details.
        model_path: Remote path to model (e.g., s3://bucket/path/to/model), or a
            `RunOutput`/`ArtifactValue` resolved at deploy time.
        model_hf_path: Hugging Face path to model (e.g., Qwen/Qwen3-0.6B).
        model_id: Model id that is exposed by SGLang.
        stream_model: When `model_path` is set, use True to stream weights from object
            storage to the GPU (Flyte loader integration). Ignored for `model_hf_path`-only apps,
            which use SGLang's normal Hugging Face download path. If False with `model_path`,
            the model is downloaded to the local filesystem first, then loaded. Also ignored when
            a draft model is configured, or when `router` is True -- see below.
        draft_model_path: Remote path to the draft model (speculator) used for speculative
            decoding, or a `RunOutput`/`ArtifactValue` resolved at deploy time. The weights are
            downloaded alongside the target model and passed as `--speculative-draft-model-path`.
            Requires `speculative_config`.
        draft_model_hf_path: Hugging Face path to the draft model, as an alternative to
            `draft_model_path` (e.g., Qwen/Qwen3-0.6B).
        speculative_config: SGLang speculative decoding configuration. Each key becomes a
            `--speculative-<key>` server arg, so `{"algorithm": "EAGLE3", "num_draft_tokens": 16}`
            renders as `--speculative-algorithm EAGLE3 --speculative-num-draft-tokens 16`. The
            draft model path is filled in from `draft_model_path`/`draft_model_hf_path` and must
            not be set here. Note that SGLang auto-selects `num_steps`, `eagle_topk` and
            `num_draft_tokens` per model family; override them only after reading acceptance
            length off `/metrics`. See
            https://docs.sglang.io/advanced_features/speculative_decoding.html.
        router: Serve behind SGLang's cache-aware router (`sglang_router.launch_server`), which
            runs data-parallel workers in one process and routes each request to the worker most
            likely to already hold its prefix. Worker counts and routing policy are ordinary
            server args, e.g. `extra_args=["--dp-size", "4", "--router-policy", "cache_aware"]`.
    """

    port: int | Port = 8080
    type: str = "SGLang"
    extra_args: str | list[str] = ""
    model_path: str | RunOutput | ArtifactValue = ""
    model_hf_path: str = ""
    model_id: str = ""
    stream_model: bool = True
    draft_model_path: str | RunOutput | ArtifactValue = ""
    draft_model_hf_path: str = ""
    speculative_config: Optional[dict[str, Any]] = None
    router: bool = False
    image: str | Image | Literal["auto"] = DEFAULT_SGLANG_IMAGE
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
            raise ValueError("server function cannot be set for SGLangAppEnvironment")

        if self._on_startup is not None:
            raise ValueError("on_startup function cannot be set for SGLangAppEnvironment")

        if self._on_shutdown is not None:
            raise ValueError("on_shutdown function cannot be set for SGLangAppEnvironment")

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
            raise ValueError("args cannot be set for SGLangAppEnvironment. Use `extra_args` to add extra arguments.")

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = list(self.extra_args)

        speculative_args = self._speculative_args()

        # Flyte blob streaming requires ``model_path`` (remote / RunOutput). HF-only apps use
        # SGLang's default loading regardless of ``stream_model``.
        #
        # Two other configurations rule streaming out:
        #   * A draft model. The loader monkeypatch is process-wide and
        #     ``FLYTE_MODEL_LOADER_REMOTE_MODEL_PATH``/``..._LOCAL_MODEL_PATH`` are a single global
        #     pair, so it can only ever describe one set of weights -- with two models loaded it
        #     would feed the target's tensors to the draft model.
        #   * ``router=True``. The router launches its workers as separate processes that never go
        #     through ``sglang-fserve``, so the patched loader is simply not installed in the
        #     processes that load weights.
        streaming_blocked_by = ""
        if self._has_draft_model:
            streaming_blocked_by = "a draft model"
        elif self.router:
            streaming_blocked_by = "router=True"

        use_flyte_blob_streaming = bool(self.stream_model and self.model_path and not streaming_blocked_by)
        if self.stream_model and self.model_path and streaming_blocked_by:
            logger.warning(
                "stream_model is not supported alongside %s: the Flyte streaming loader is installed "
                "process-wide for a single set of weights. Falling back to downloading the model(s) to "
                "the container filesystem.",
                streaming_blocked_by,
            )

        entrypoint = ["python", "-m", "sglang_router.launch_server"] if self.router else ["sglang-fserve"]

        # SGLang binds 127.0.0.1 by default, which is unreachable from outside the container.
        host_args = [] if "--host" in extra_args else ["--host", "0.0.0.0"]

        self.args = _shell_safe(
            [
                *entrypoint,
                "--model-path",
                self._model_mount_path,
                "--served-model-name",
                self.model_id,
                *host_args,
                "--port",
                str(self.get_port().port),
                *speculative_args,
                *extra_args,
            ]
        )

        if self.parameters:
            raise ValueError("parameters cannot be set for SGLangAppEnvironment")

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
        self.links = [flyte.app.Link(path="/docs", title="SGLang OpenAPI Docs", is_relative=True), *self.links]

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_SGLANG_IMAGE

        super().__post_init__()

    @property
    def _has_draft_model(self) -> bool:
        return bool(self.draft_model_path or self.draft_model_hf_path)

    def _speculative_args(self) -> list[str]:
        """Render the flat `--speculative-*` flags, pointing at the draft model when there is one."""
        if self._has_draft_model and self.speculative_config is None:
            raise ValueError(
                "speculative_config must be defined when a draft model is set. It selects the "
                'speculative decoding algorithm and its knobs, e.g. {"algorithm": "EAGLE3"}.'
            )

        if self.speculative_config is None:
            return []

        config = dict(self.speculative_config)
        if "draft_model_path" in config or "draft-model-path" in config:
            raise ValueError(
                "speculative_config cannot set 'draft_model_path'. Use the draft_model_path or "
                "draft_model_hf_path field instead, so that the weights are mounted into the container."
            )
        if self._has_draft_model:
            config["draft_model_path"] = self.draft_model_hf_path or self._draft_model_mount_path

        args: list[str] = []
        for key, value in config.items():
            # Both "algorithm" and the fully spelled out "speculative_algorithm" are accepted.
            name = key.replace("_", "-").removeprefix("speculative-")
            flag = f"--speculative-{name}"
            if isinstance(value, bool):
                # Flag-style options carry no value; a False one is simply left out.
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])
        return args

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
        router = kwargs.pop("router", None)

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
        if router is not None:
            kwargs["router"] = router
        return replace(self, **kwargs)
