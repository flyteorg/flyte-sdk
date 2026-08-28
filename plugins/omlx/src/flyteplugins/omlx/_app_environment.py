from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte.app._types import Port
from flyte.models import SerializationContext

from flyteplugins.omlx._constants import DEFAULT_OMLX_PORT, OMLX_MIN_VERSION_STR

# omlx is darwin/arm64 only, so the "default image" here is mostly a placeholder
# that lets remote-deploy scaffolding succeed. In practice, this plugin is
# meant to be driven by `flyte serve --local` on an Apple Silicon Mac with
# `omlx` already installed. See the plugin README for details.
DEFAULT_OMLX_IMAGE = (
    flyte.Image.from_debian_base(name="omlx-app-image", install_flyte=False)
    .with_pip_packages("flyteplugins-omlx", pre=True)
    .with_pip_packages(f"omlx>={OMLX_MIN_VERSION_STR}")
)


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class OMLXAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by [oMLX](https://github.com/jundot/omlx) for serving
    LLMs on Apple Silicon Macs.

    oMLX is an OpenAI-compatible inference server built on Apple's MLX framework.
    It is **macOS + Apple Silicon only**. The recommended way to use this plugin
    is via local serving:

    ```bash
    flyte serve --local examples/genai/omlx/omlx_app.py omlx_app
    ```

    The user is responsible for installing `omlx` on the host machine. The
    easiest way is `pip install 'flyteplugins-omlx[omlx]'` (which only resolves
    on darwin-arm64), or follow the upstream install instructions:
    https://github.com/jundot/omlx#installation.

    :param name: The name of the application.
    :param image: Container image. Mostly a placeholder for this plugin
        because oMLX cannot run inside Linux containers — leave as default for
        local serving.
    :param port: Port the oMLX server listens on. Defaults to 8000 (oMLX's own
        default).
    :param resources: Compute resources for the app (used only for remote
        deploy scaffolding).
    :param secrets: Secrets requested by the app.
    :param env_vars: Environment variables for the server process.
    :param scaling: Scaling configuration (remote deploy only).
    :param domain: Custom domain for the deployed app.
    :param cluster_pool: Target cluster pool for remote deploy.
    :param requires_auth: Whether the public URL requires Flyte auth.
    :param type: App type tag.
    :param model_dir: Local directory of MLX models, passed to
        ``omlx serve --model-dir``. Each subdirectory is exposed as a separate
        ``model_id``. If unset, oMLX falls back to ``~/.omlx/models``.
    :param model_id: Optional friendly id used for the served model. This is
        not passed to oMLX (oMLX derives ids from ``--model-dir`` subdirectories
        or HuggingFace repo ids), but it shows up in the Flyte UI and in
        OpenAI client requests as the ``model`` parameter.
    :param api_key: Optional API key for the oMLX server (sent through
        ``--api-key``). For real secrets, prefer ``secrets`` + ``env_vars``.
    :param extra_args: Extra CLI args appended to ``omlx serve``. See
        ``omlx serve --help`` for the full list (``--max-model-memory``,
        ``--max-concurrent-requests``, ``--paged-ssd-cache-dir``,
        ``--hf-endpoint``, ``--log-level``, etc.).
    """

    port: int | Port = DEFAULT_OMLX_PORT
    type: str = "oMLX"
    extra_args: str | list[str] = ""
    model_dir: str = ""
    model_id: str = ""
    api_key: str = ""
    image: str | Image | Literal["auto"] = DEFAULT_OMLX_IMAGE
    _resolved_args: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

        if self._server is not None:
            raise ValueError("server function cannot be set for OMLXAppEnvironment")

        if self._on_startup is not None:
            raise ValueError("on_startup function cannot be set for OMLXAppEnvironment")

        if self._on_shutdown is not None:
            raise ValueError("on_shutdown function cannot be set for OMLXAppEnvironment")

        if self.args:
            raise ValueError("args cannot be set for OMLXAppEnvironment. Use `extra_args` to add extra arguments.")

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = list(self.extra_args)

        # We invoke the `flyte-omlx` wrapper (defined in this package) instead
        # of `omlx` directly so the user gets an actionable install hint if
        # oMLX is missing on the host. When oMLX is installed, the wrapper
        # `execv`s straight into it.
        cmd: list[str] = [
            "flyte-omlx",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.get_port().port),
        ]
        if self.model_dir:
            cmd += ["--model-dir", self.model_dir]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        cmd += extra_args

        self.args = cmd
        self._resolved_args = cmd

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_OMLX_IMAGE

        # Helpful default link to the OpenAI-compatible models endpoint so
        # users can sanity-check the server from the Flyte UI.
        self.links = [
            flyte.app.Link(path="/v1/models", title="oMLX Models", is_relative=True),
            *self.links,
        ]

        super().__post_init__()

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        """Return the container arguments for oMLX."""
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
    ) -> OMLXAppEnvironment:
        port = kwargs.pop("port", None)
        extra_args = kwargs.pop("extra_args", None)
        model_dir = kwargs.pop("model_dir", None)
        model_id = kwargs.pop("model_id", None)
        api_key = kwargs.pop("api_key", None)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        kwargs = self._get_kwargs()
        kwargs["name"] = name
        # Reset args so __post_init__ rebuilds them with the new fields.
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
        if model_dir is not None:
            kwargs["model_dir"] = model_dir
        if model_id is not None:
            kwargs["model_id"] = model_id
        if api_key is not None:
            kwargs["api_key"] = api_key
        return replace(self, **kwargs)
