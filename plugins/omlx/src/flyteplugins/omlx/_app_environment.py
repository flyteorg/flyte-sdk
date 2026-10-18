from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte.app._types import Port
from flyte.models import SerializationContext

from flyteplugins.omlx._constants import DEFAULT_OMLX_PORT


def _shell_safe(args: list[str]) -> list[str]:
    """Quote args that have to survive a trip through a shell.

    `fserve` runs the app with `Popen(" ".join(args), shell=True)` (flyte/_bin/serve.py),
    so any token carrying spaces or quotes -- a `--model-dir` under a user name with a
    space, say -- reaches the server mangled unless it is quoted here. `shlex.quote` is
    the identity function for ordinary tokens, so this is a no-op for everything else.

    Tokens starting with `$` are left alone: `fserve` expands those against the
    environment *before* joining, and quoting would turn the marker into a literal.
    """
    return [arg if arg.startswith("$") else shlex.quote(arg) for arg in args]


# oMLX runs only on macOS with Apple Silicon: `mlx` has no Linux wheels, so this default
# image only installs the plugin. The `omlx` binary itself is expected on the Mac's host
# (Homebrew or pip-from-git), where the devbox's Apple Silicon bridge runs the app; the
# `flyte-omlx` wrapper resolves it (probing Homebrew paths that host-execution's PATH
# omits). See the plugin README.
DEFAULT_OMLX_IMAGE = flyte.Image.from_debian_base(name="omlx-app-image").with_pip_packages(
    "flyteplugins-omlx",
    pre=True,
)


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class OMLXAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by [oMLX](https://github.com/jundot/omlx) for serving
    LLMs on Apple Silicon Macs.

    oMLX is an OpenAI-compatible inference server built on Apple's MLX framework.
    It is **macOS + Apple Silicon only**. On a flyte devbox started with
    ``--apple-silicon``, this environment requests ``gpu="AppleSilicon"`` by default,
    so the devbox's bridge runs ``omlx serve`` natively on the Mac (in an environment
    built from the image's pip packages) and forwards the app's URL, port and logs.

    ```bash
    flyte start devbox --apple-silicon
    flyte create config --devbox
    python examples/accelerators/omlx_serve.py
    ```

    The user is responsible for installing `omlx` on the host machine:

    ```bash
    brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx
    # or: pip install git+https://github.com/jundot/omlx
    ```

    :param name: The name of the application.
    :param image: Container image. oMLX cannot run inside Linux containers, so this
        mostly describes the pip *packages* that get rebuilt into the Mac's host
        environment. Leave as default for the devbox workflow; you typically only
        override it to add pip packages (e.g. ``with_pip_packages("omlx @ git+...")``).
    :param port: Port the oMLX server listens on. Defaults to 8000 (oMLX's own default).
    :param resources: Compute resources. Defaults to ``gpu="AppleSilicon"`` so the app
        runs on the Mac's GPU via the devbox bridge; other GPU types are rejected.
    :param secrets: Secrets requested by the app.
    :param env_vars: Environment variables for the server process.
    :param scaling: Scaling configuration.
    :param domain: Custom domain for the deployed app.
    :param cluster_pool: Target cluster pool.
    :param requires_auth: Whether the public URL requires Flyte auth.
    :param type: App type tag.
    :param model_dir: Local directory of MLX models, passed to ``omlx serve --model-dir``.
        Each subdirectory is exposed as a separate ``model_id``. If unset, oMLX falls
        back to ``~/.omlx/models``.
    :param model_id: Optional friendly id surfaced in the Flyte UI and usable as the
        ``model`` in OpenAI client requests. oMLX derives actual served ids from
        ``--model-dir`` subdirectories or HuggingFace repo ids.
    :param api_key: Optional API key for the oMLX server (sent through ``--api-key``).
        For real secrets, prefer ``secrets`` + ``env_vars``.
    :param extra_args: Extra CLI args appended to ``omlx serve``. See ``omlx serve
        --help`` for the full list (``--max-model-memory``,
        ``--max-concurrent-requests``, ``--paged-ssd-cache-dir``, ``--hf-endpoint``,
        ``--log-level``, etc.).
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

        # oMLX is mac-only: default to running natively on the Mac's GPU through the
        # devbox's Apple Silicon bridge, and refuse any other accelerator.
        device = self.resources.get_device() if self.resources is not None else None
        if device is None:
            if self.resources is None:
                self.resources = Resources(gpu="AppleSilicon")
            else:
                self.resources = replace(self.resources, gpu="AppleSilicon")
        elif device.device_class != "APPLE_SILICON":
            raise ValueError(
                "OMLXAppEnvironment only runs on Apple Silicon (gpu='AppleSilicon') because oMLX uses MLX, "
                "which is darwin/arm64 only; got resources with a non-Apple Silicon device."
            )

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = list(self.extra_args)

        # oMLX is served locally on the Mac (devbox ports are bound to 127.0.0.1), so bind
        # loopback by default; a --host in extra_args wins.
        host_args = [] if any(a == "--host" for a in extra_args) else ["--host", "127.0.0.1"]

        # We invoke the `flyte-omlx` wrapper (defined in this package) instead of `omlx`
        # directly so the user gets an actionable install hint if oMLX is missing on the
        # host. When oMLX is installed, the wrapper `execv`s straight into it.
        cmd: list[str] = [
            "flyte-omlx",
            "serve",
            *host_args,
            "--port",
            str(self.get_port().port),
        ]
        if self.model_dir:
            cmd += ["--model-dir", self.model_dir]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        cmd += extra_args

        self.args = _shell_safe(cmd)
        self._resolved_args = cmd

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_OMLX_IMAGE

        # Helpful default link to the OpenAI-compatible models endpoint so users can
        # sanity-check the server from the Flyte UI.
        self.links = [flyte.app.Link(path="/v1/models", title="oMLX Models", is_relative=True), *self.links]

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
