from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte.app._types import Port
from flyte.models import SerializationContext

from flyteplugins.rapidmlx._constants import DEFAULT_RAPID_MLX_PORT


def _shell_safe(args: list[str]) -> list[str]:
    """Quote args that have to survive a trip through a shell.

    `fserve` runs the app with `Popen(" ".join(args), shell=True)` (flyte/_bin/serve.py),
    so any token carrying spaces or quotes -- a model argument with a
    space, say -- reaches the server mangled unless it is quoted here. `shlex.quote` is
    the identity function for ordinary tokens, so this is a no-op for everything else.

    Tokens starting with `$` are left alone: `fserve` expands those against the
    environment *before* joining, and quoting would turn the marker into a literal.
    """
    return [arg if arg.startswith("$") else shlex.quote(arg) for arg in args]


# The default image installs the plugin; the Apple Silicon bridge runs the
# server natively on the Mac, where Rapid-MLX must already be installed.
DEFAULT_RAPID_MLX_IMAGE = flyte.Image.from_debian_base(name="rapidmlx-app-image").with_pip_packages(
    "flyteplugins-rapidmlx",
    pre=True,
)


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class RapidMLXAppEnvironment(flyte.app.AppEnvironment):
    """Serve LLMs on Apple Silicon with [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX).

    Rapid-MLX is an OpenAI-compatible local LLM inference server built on MLX.
    With `flyte start devbox --apple-silicon`, the bridge runs `rapid-mlx serve`
    natively on the Mac and forwards the app's URL, port and logs.

    Install the server on the Mac with `brew install rapid-mlx` or
    `pip install rapid-mlx` (Python 3.12+ for the server).

    ```python
    from flyteplugins.rapidmlx import RapidMLXAppEnvironment

    app = RapidMLXAppEnvironment(name="rapidmlx-local", model_id="qwen3.5-4b-4bit")
    ```

    Args:
        name: The application name.
        image: Image whose pip packages are rebuilt for the Mac's host environment.
            The default installs only this plugin; install Rapid-MLX on the host.
        port: Server port. Defaults to 8000, giving `http://localhost:8000/v1`.
            The plugin passes an explicit port, so an occupied port fails without fallback.
        resources: Defaults to `gpu="AppleSilicon"`; other GPU types are rejected.
        secrets: Secrets requested by the app.
        env_vars: Server environment variables. Supports `RAPID_MLX_API_KEY` for auth.
        scaling: Scaling configuration.
        domain: Custom domain for the deployed app.
        cluster_pool: Target cluster pool.
        requires_auth: Whether the public URL requires Flyte auth.
        type: App type tag.
        model_id: MLX model id or alias passed as the positional model argument.
            Defaults to `qwen3.5-4b-4bit`. Use `/v1/models` to discover the served id
            for OpenAI client requests.
        api_key: Optional `--api-key`. Prefer secrets with `env_vars` for real keys.
        extra_args: Extra arguments appended to `rapid-mlx serve`, as a list or
            shell-split string. See `rapid-mlx serve --help`.
    """

    port: int | Port = DEFAULT_RAPID_MLX_PORT
    type: str = "Rapid-MLX"
    extra_args: str | list[str] = ""
    model_id: str = "qwen3.5-4b-4bit"
    api_key: str = ""
    image: str | Image | Literal["auto"] = DEFAULT_RAPID_MLX_IMAGE
    _resolved_args: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

        if self._server is not None:
            raise ValueError("server function cannot be set for RapidMLXAppEnvironment")

        if self._on_startup is not None:
            raise ValueError("on_startup function cannot be set for RapidMLXAppEnvironment")

        if self._on_shutdown is not None:
            raise ValueError("on_shutdown function cannot be set for RapidMLXAppEnvironment")

        if self.args:
            raise ValueError("args cannot be set for RapidMLXAppEnvironment. Use `extra_args` to add extra arguments.")

        # Rapid-MLX is mac-only: default to running natively on the Mac's GPU through the
        # devbox's Apple Silicon bridge, and refuse any other accelerator.
        device = self.resources.get_device() if self.resources is not None else None
        if device is None:
            if self.resources is None:
                self.resources = Resources(gpu="AppleSilicon")
            else:
                self.resources = replace(self.resources, gpu="AppleSilicon")
        elif device.device_class != "APPLE_SILICON":
            raise ValueError(
                "RapidMLXAppEnvironment only runs on Apple Silicon (gpu='AppleSilicon') because Rapid-MLX uses MLX, "
                "which is darwin/arm64 only; got resources with a non-Apple Silicon device."
            )

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = list(self.extra_args)

        # Rapid-MLX is served locally on the Mac (devbox ports are bound to 127.0.0.1), so bind
        # loopback by default; a --host in extra_args wins.
        host_args = [] if any(a == "--host" for a in extra_args) else ["--host", "127.0.0.1"]

        # We invoke the `flyte-rapidmlx` wrapper (defined in this package) instead of `rapid-mlx`
        # directly so the user gets an actionable install hint if Rapid-MLX is missing on the
        # host. When Rapid-MLX is installed, the wrapper `execv`s straight into it.
        cmd: list[str] = [
            "flyte-rapidmlx",
            "serve",
            self.model_id,
            *host_args,
            "--port",
            str(self.get_port().port),
        ]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        cmd += extra_args

        self.args = _shell_safe(cmd)
        self._resolved_args = cmd

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_RAPID_MLX_IMAGE

        # Helpful default link to the OpenAI-compatible models endpoint so users can
        # sanity-check the server from the Flyte UI.
        self.links = [flyte.app.Link(path="/v1/models", title="Rapid-MLX Models", is_relative=True), *self.links]

        super().__post_init__()

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        """Return the container arguments for Rapid-MLX."""
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
        service_account: Optional[str] = None,
        **kwargs: Any,
    ) -> RapidMLXAppEnvironment:
        port = kwargs.pop("port", None)
        extra_args = kwargs.pop("extra_args", None)
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
        if service_account is not None:
            kwargs["service_account"] = service_account
        if port is not None:
            kwargs["port"] = port
        if extra_args is not None:
            kwargs["extra_args"] = extra_args
        if model_id is not None:
            kwargs["model_id"] = model_id
        if api_key is not None:
            kwargs["api_key"] = api_key
        return replace(self, **kwargs)
