from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte.app import ArtifactValue, Parameter, RunOutput
from flyte.app._types import Port
from flyte.models import SerializationContext

from flyteplugins.llamacpp._image import DEFAULT_LLAMA_CPP_IMAGE


def _shell_safe(args: list[str]) -> list[str]:
    """Quote args that have to survive a trip through a shell.

    `fserve` runs the app with `Popen(" ".join(args), shell=True)` (flyte/_bin/serve.py), so
    any token carrying spaces or quotes -- a `--chat-template` blob, say -- reaches the server
    mangled unless it is quoted here. `shlex.quote` is the identity function for ordinary
    tokens, so this is a no-op for everything else.

    Tokens starting with `$` are left alone: `fserve` expands those against the container
    environment *before* joining, and quoting would turn the marker into a literal.
    """
    return [arg if arg.startswith("$") else shlex.quote(arg) for arg in args]


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class LlamaCppAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by llama.cpp (llama-server) for serving GGUF models.

    This environment serves an OpenAI-compatible endpoint (under `/v1`) plus the llama.cpp
    Web UI, with the specified GGUF model and configuration. llama.cpp shines where vLLM and
    SGLang don't fit: quantized GGUF weights, partial CPU offload of models larger than VRAM,
    and CPU-only serving.

    Args:
        name: The name of the application.
        port: Port the application listens on. Defaults to 8080.
        requests: Compute resource requests for application.
        secrets: Secrets that are requested for application.
        limits: Compute resource limits for application.
        env_vars: Environment variables to set for the application.
        scaling: Scaling configuration for the app environment.
        domain: Domain to use for the app.
        cluster_pool: The target cluster_pool where the app should be deployed.
        requires_auth: Whether the public URL requires authentication.
        type: Type of app.
        extra_args: Extra args to pass to `llama-server`, e.g. `"--ctx-size 32768 --jinja"`.
            Run `llama-server --help` or see
            https://github.com/ggml-org/llama.cpp/tree/master/tools/server for details.
        model_path: Remote path to the GGUF weights -- a directory containing `.gguf` file(s)
            or a direct path to one (e.g. s3://bucket/path/to/model), or a
            `RunOutput`/`ArtifactValue` resolved at deploy time. The weights are downloaded
            into the container and the served `.gguf` is located at startup (for sharded
            models, the `-00001-of-` shard is picked; llama-server finds the rest).
        model_hf_path: Hugging Face GGUF repo, optionally with a quant tag (e.g.
            `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M`). Passed to llama-server as `--hf-repo`,
            which downloads the weights at startup.
        model_id: Model id exposed by the server (llama-server's `--alias`).
        draft_model_path: Remote path to the draft model GGUF used for speculative decoding,
            or a `RunOutput`/`ArtifactValue` resolved at deploy time. Downloaded alongside the
            target model and passed as `--model-draft`. Tune the speculation via `extra_args`
            (`--draft-max`, `--draft-min`, `--gpu-layers-draft`, ...).
        draft_model_hf_path: Hugging Face GGUF repo for the draft model, as an alternative to
            `draft_model_path`. Passed as `--hf-repo-draft`.
    """

    port: int | Port = 8080
    type: str = "llama.cpp"
    extra_args: str | list[str] = ""
    model_path: str | RunOutput | ArtifactValue = ""
    model_hf_path: str = ""
    model_id: str = ""
    draft_model_path: str | RunOutput | ArtifactValue = ""
    draft_model_hf_path: str = ""
    image: str | Image | Literal["auto"] = DEFAULT_LLAMA_CPP_IMAGE
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
            raise ValueError("server function cannot be set for LlamaCppAppEnvironment")

        if self._on_startup is not None:
            raise ValueError("on_startup function cannot be set for LlamaCppAppEnvironment")

        if self._on_shutdown is not None:
            raise ValueError("on_shutdown function cannot be set for LlamaCppAppEnvironment")

        if self.model_id == "":
            raise ValueError("model_id must be defined")

        if self.model_path == "" and self.model_hf_path == "":
            raise ValueError("model_path or model_hf_path must be defined")
        if self.model_path != "" and self.model_hf_path != "":
            raise ValueError("model_path and model_hf_path cannot be set at the same time")

        if self.draft_model_path != "" and self.draft_model_hf_path != "":
            raise ValueError("draft_model_path and draft_model_hf_path cannot be set at the same time")

        if self.args:
            raise ValueError("args cannot be set for LlamaCppAppEnvironment. Use `extra_args` to add extra arguments.")

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = list(self.extra_args)

        # The GGUF filename inside a mounted directory is unknown at deploy time, so mounted
        # weights go through the `llama-cpp-fserve` shim, which resolves `--model-dir` /
        # `--draft-model-dir` to concrete .gguf paths and execs llama-server.
        if self.model_path:
            model_args = ["--model-dir", self._model_mount_path]
        else:
            model_args = ["--hf-repo", self.model_hf_path]

        draft_args: list[str] = []
        if self.draft_model_path:
            draft_args = ["--draft-model-dir", self._draft_model_mount_path]
        elif self.draft_model_hf_path:
            draft_args = ["--hf-repo-draft", self.draft_model_hf_path]

        # llama-server binds 127.0.0.1 by default, which is unreachable from outside the
        # container.
        host_args = [] if "--host" in extra_args else ["--host", "0.0.0.0"]

        self.args = _shell_safe(
            [
                "llama-cpp-fserve",
                *model_args,
                "--alias",
                self.model_id,
                *host_args,
                "--port",
                str(self.get_port().port),
                *draft_args,
                *extra_args,
            ]
        )

        if self.parameters:
            raise ValueError("parameters cannot be set for LlamaCppAppEnvironment")

        parameters: list[Parameter] = []
        if self.model_path:
            parameters.append(
                Parameter(
                    name="model_path",
                    value=self.model_path,
                    download=True,
                    mount=self._model_mount_path,
                )
            )
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

        self.links = [flyte.app.Link(path="/", title="llama.cpp Web UI", is_relative=True), *self.links]

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_LLAMA_CPP_IMAGE

        super().__post_init__()

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        """Return the container arguments for llama.cpp."""
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
    ) -> LlamaCppAppEnvironment:
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
        model_id = kwargs.pop("model_id", None)

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
        if model_id is not None:
            kwargs["model_id"] = model_id
        return replace(self, **kwargs)
