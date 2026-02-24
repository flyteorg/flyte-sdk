import datetime
import hashlib
import json
import logging
import re
import tempfile
import textwrap
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import flyte
from flyte.errors import InvalidPackageError
from flyte.io import Dir, File
from flyte.syncify import syncify

if TYPE_CHECKING:
    from flyte.extras._container import ContainerTask

logger = logging.getLogger(__name__)

# Types that can be declared as sandbox inputs or outputs.
# Anything outside this set is rejected at create() time.
_SUPPORTED_TYPES: frozenset[type] = frozenset({int, float, str, bool, datetime.datetime, datetime.timedelta, File, Dir})

sandbox_environment = flyte.TaskEnvironment(
    name="sandbox-runtime",
    image=flyte.Image.from_debian_base(install_flyte=False),
)


@dataclass
class ImageConfig:
    """Configuration for Docker image building at runtime."""

    registry: Optional[str] = None
    registry_secret: Optional[str] = None
    python_version: Optional[tuple[int, int]] = None


@dataclass
class _Sandbox:
    # Identity
    name: Optional[str] = None

    # Task spec — code mode (Python script with argparse-style I/O)
    code: Optional[str] = None
    inputs: dict[str, type] = field(default_factory=dict)
    outputs: dict[str, type] = field(default_factory=dict)

    # Task spec — command mode (arbitrary command, e.g. pytest runner)
    command: Optional[list[str]] = None
    arguments: Optional[list[str]] = None

    # Image
    packages: list[str] = field(default_factory=list)
    system_packages: list[str] = field(default_factory=list)
    additional_commands: list[str] = field(default_factory=list)
    image_config: Optional[ImageConfig] = None
    image_name: Optional[str] = None
    image: Optional[str] = None  # If provided, skip build and use this image directly

    # Python code behaviour
    auto_io: bool = True  # True = auto-inject typed I/O; False = verbatim Python script

    # Runtime
    resources: Optional[flyte.Resources] = None
    block_network: bool = True
    retries: int = 0
    timeout: Optional[int] = None
    env_vars: Optional[dict[str, str]] = None
    secrets: Optional[list] = None
    cache: str = "auto"

    def _task_name(self) -> str:
        return self.name or f"sandbox-{flyte.ctx().action.name}"

    def _create_image_spec(self) -> flyte.Image:
        spec_name = self.image_name or self._default_image_name()
        config = self.image_config or ImageConfig()

        image = flyte.Image.from_debian_base(
            install_flyte=False,
            registry=config.registry,
            registry_secret=config.registry_secret,
            python_version=config.python_version,
            name=spec_name,
        )

        apt_packages = list(self.system_packages)
        if "gcc" not in apt_packages:
            apt_packages.extend(["gcc", "g++", "make"])
        if apt_packages:
            image = image.with_apt_packages(*apt_packages)

        if self.packages:
            image = image.with_pip_packages(*self.packages)

        if self.additional_commands:
            image = image.with_commands(self.additional_commands)

        return image

    @syncify
    async def _build(self) -> str:
        try:
            result = await flyte.build.aio(self._create_image_spec())
            return result.uri
        except Exception as e:
            error_msg = str(e)
            if "Unable to locate package" in error_msg or "has no installation candidate" in error_msg:
                match = re.search(r"(?:Unable to locate package|Package '?)([\w.+-]+)", error_msg)
                if match:
                    raise InvalidPackageError(match.group(1), error_msg) from e
            raise

    def _generate_auto_script(self) -> str:
        """Wrap user code with auto-generated argparse preamble and output epilogue.

        The generated script:
        1. Parses all declared inputs from CLI args (``--name value``).
        2. Executes the user's code verbatim with those names in scope.
        3. Writes each declared scalar output variable to ``/var/outputs/<name>``.

        File / Dir inputs are injected as path strings.
        File / Dir outputs must be written by user code to ``/var/outputs/<name>``.
        """
        scalar_inputs = {k: v for k, v in self.inputs.items() if v not in (File, Dir)}
        io_inputs = {k: v for k, v in self.inputs.items() if v in (File, Dir)}
        scalar_outputs = {k: v for k, v in self.outputs.items() if v not in (File, Dir)}

        lines: list[str] = ["import argparse as _ap_", "import pathlib as _pl_"]

        # preamble: parse inputs
        if scalar_inputs or io_inputs:
            lines += ["", "_parser = _ap_.ArgumentParser(add_help=False)"]
            for name, typ in scalar_inputs.items():
                if typ is bool:
                    lines.append(
                        f"_parser.add_argument('--{name}', type=lambda _v: _v.lower() not in ('false', '0', ''))"
                    )
                elif typ in (int, float, str):
                    lines.append(f"_parser.add_argument('--{name}', type={typ.__name__})")
                else:
                    # datetime.datetime / datetime.timedelta — receive as string
                    lines.append(f"_parser.add_argument('--{name}', type=str)")
            for name in io_inputs:
                lines.append(f"_parser.add_argument('--{name}', type=str)")
            lines.append("_args = _parser.parse_args()")

            for name, typ in scalar_inputs.items():
                if issubclass(typ, datetime.datetime):
                    lines += [
                        "import datetime as _dt_",
                        f"{name} = _dt_.datetime.fromisoformat(_args.{name})",
                    ]
                elif issubclass(typ, datetime.timedelta):
                    # Supports:
                    # - Local execution: str(timedelta)
                    #       "1 day, 0:00:00"
                    #       "0:01:30"
                    #       "2 days, 3:04:05.123456"
                    #
                    # - Protobuf TextFormat (google.protobuf.Duration)
                    #       "seconds:86400"
                    #       "seconds:1 nanos:500000000"

                    lines += [
                        "import datetime as _dt_, re as _re_",
                        # Regex for Python str(timedelta)
                        "_TD_PY_RE = _re_.compile("
                        r"'(?:(\\d+) days?, )?(?:(\\d+):)?(\\d+):(\\d+)(?:\\.(\\d+))?'"
                        ")",
                        "def _parse_timedelta(_s):",
                        "    if _s is None:",
                        "        return None",
                        "    # --- Protobuf TextFormat: seconds / nanos ---",
                        "    if _s.startswith('seconds:'):",
                        "        parts = dict(p.split(':', 1) for p in _s.split() if ':' in p)",
                        "        sec = int(parts.get('seconds', 0))",
                        "        nanos = int(parts.get('nanos', 0))",
                        "        return _dt_.timedelta(",
                        "            seconds=sec,",
                        "            microseconds=nanos // 1000,",
                        "        )",
                        "    # --- Python str(timedelta) ---",
                        "    m = _TD_PY_RE.match(_s)",
                        "    if m:",
                        "        return _dt_.timedelta(",
                        "            days=int(m.group(1) or 0),",
                        "            hours=int(m.group(2) or 0),",
                        "            minutes=int(m.group(3) or 0),",
                        "            seconds=int(m.group(4) or 0),",
                        "            microseconds=int((m.group(5) or '0').ljust(6, '0')[:6]),",
                        "        )",
                        "    raise ValueError(f'Unsupported timedelta format: {_s!r}')",
                        f"{name} = _parse_timedelta(_args.{name})",
                    ]
                else:
                    lines.append(f"{name} = _args.{name}")
            for name in io_inputs:
                lines.append(f"{name} = _args.{name}  # path to bound file/directory")

        # user code
        lines += ["", "# user code", ""]
        lines.append(textwrap.dedent(self.code).strip())

        # epilogue: write scalar outputs
        if scalar_outputs:
            lines += ["", "# auto-generated output writing", ""]
            lines.append("_out_ = _pl_.Path('/var/outputs')")
            lines.append("_out_.mkdir(parents=True, exist_ok=True)")
            for name, typ in scalar_outputs.items():
                if issubclass(typ, datetime.datetime):
                    lines.append(f"(_out_ / '{name}').write_text({name}.isoformat())")
                else:
                    lines.append(f"(_out_ / '{name}').write_text(str({name}))")

        return "\n".join(lines) + "\n"

    def _make_container_task(self, image: str, task_name: str) -> "ContainerTask":
        from flyte.extras._container import ContainerTask

        extra_kwargs: dict[str, Any] = {}
        if self.timeout is not None:
            extra_kwargs["timeout"] = self.timeout
        if self.env_vars is not None:
            extra_kwargs["env_vars"] = self.env_vars
        if self.secrets is not None:
            extra_kwargs["secrets"] = self.secrets

        resources = self.resources or flyte.Resources(cpu=1, memory="1Gi")

        if self.code is not None:
            if not self.auto_io:
                # Verbatim mode: run the Python script as-is with no injected CLI args.
                # The user is fully responsible for reading inputs and writing outputs.
                bash_cmd = (
                    "set -o pipefail && python $1; "
                    "_exit=$?; mkdir -p /var/outputs; echo $_exit > /var/outputs/exit_code"
                )
                return ContainerTask(
                    name=task_name,
                    image=image,
                    input_data_dir="/var/inputs",
                    output_data_dir="/var/outputs",
                    inputs={**self.inputs, "_script": File},
                    outputs=dict(self.outputs),
                    command=["/bin/bash", "-c", bash_cmd],
                    arguments=["/bin/bash", "/var/inputs/_script"],
                    resources=resources,
                    block_network=self.block_network,
                    retries=self.retries,
                    cache=self.cache,
                    **extra_kwargs,
                )

            # Auto-inject mode: generate CLI args for inputs; the generated script
            # parses them via argparse and writes scalar outputs automatically.
            cli_args = []
            arguments = ["/bin/bash", "/var/inputs/_script"]
            positional_index = 2

            for arg_name, arg_type in self.inputs.items():
                if arg_type in (File, Dir):
                    cli_args.extend([f"--{arg_name}", f"${positional_index}"])
                    arguments.append(f"/var/inputs/{arg_name}")
                    positional_index += 1
                else:
                    # Single-quote the template value so bash treats it as one token
                    # even when the substituted value contains spaces (e.g. datetime
                    # "2024-01-01 00:00:00+00:00" or timedelta "1 day, 0:00:00").
                    cli_args.extend([f"--{arg_name}", f"'{{{{.inputs.{arg_name}}}}}'"])

            python_args = " ".join(cli_args)
            python_cmd = f"python $1 {python_args}" if python_args else "python $1"
            bash_cmd = (
                f"set -o pipefail && {python_cmd}; "
                f"_exit=$?; mkdir -p /var/outputs; echo $_exit > /var/outputs/exit_code"
            )

            return ContainerTask(
                name=task_name,
                image=image,
                input_data_dir="/var/inputs",
                output_data_dir="/var/outputs",
                inputs={**self.inputs, "_script": File},
                outputs=dict(self.outputs),
                command=["/bin/bash", "-c", bash_cmd],
                arguments=arguments,
                resources=resources,
                block_network=self.block_network,
                retries=self.retries,
                cache=self.cache,
                **extra_kwargs,
            )
        else:
            # Command mode: use the provided command and arguments directly.
            return ContainerTask(
                name=task_name,
                image=image,
                input_data_dir="/var/inputs",
                output_data_dir="/var/outputs",
                inputs=dict(self.inputs),
                outputs=dict(self.outputs),
                command=self.command or [],
                arguments=self.arguments or [],
                resources=resources,
                block_network=self.block_network,
                retries=self.retries,
                cache=self.cache,
                **extra_kwargs,
            )

    @syncify
    async def run(self, image: Optional[str] = None, **kwargs) -> Any:
        """Build the image (if needed) and execute the sandbox.

        Args:
            image: Pre-built image URI. If ``None``, the image is built automatically.
            **kwargs: Runtime input values matching the declared inputs.

        Returns:
            Tuple of typed outputs.
        """
        if image is None:
            image = self.image or await self._build.aio()

        task_name = self._task_name()
        task = self._make_container_task(image, task_name)
        task.parent_env = weakref.ref(sandbox_environment)
        task.parent_env_name = task_name

        if self.code is not None:
            script_content = self._generate_auto_script() if self.auto_io else self.code
            script_path = Path(tempfile.gettempdir()) / f"{task_name}_generated.py"
            script_path.write_text(script_content)
            script = await File.from_local(
                str(script_path),
                hash_method=hashlib.sha256(script_content.encode()).hexdigest(),
            )
            return await task(_script=script, **kwargs)

        return await task(**kwargs)

    def _default_image_name(self) -> str:
        spec = {
            "packages": sorted(self.packages),
            "system_packages": sorted(self.system_packages),
        }
        config_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.name or 'sandbox'}-{config_hash}"


def create(
    name: Optional[str] = None,
    *,
    code: Optional[str] = None,
    inputs: Optional[dict[str, type]] = None,
    outputs: Optional[dict[str, type]] = None,
    command: Optional[list[str]] = None,
    arguments: Optional[list[str]] = None,
    packages: Optional[list[str]] = None,
    system_packages: Optional[list[str]] = None,
    additional_commands: Optional[list[str]] = None,
    resources: Optional[flyte.Resources] = None,
    image_config: Optional[ImageConfig] = None,
    image_name: Optional[str] = None,
    image: Optional[str] = None,
    auto_io: bool = True,
    block_network: bool = True,
    retries: int = 0,
    timeout: Optional[int] = None,
    env_vars: Optional[dict[str, str]] = None,
    secrets: Optional[list] = None,
    cache: str = "auto",
) -> _Sandbox:
    """Create a stateless Python code sandbox.

    Three modes, mutually exclusive:

    - **Auto-IO mode** (``code`` provided, ``auto_io=True``, default): write
      just the business logic. Flyte auto-generates an argparse preamble so
      declared inputs are available as local variables, and writes declared
      scalar outputs to ``/var/outputs/`` automatically. No boilerplate needed.
    - **Verbatim mode** (``code`` provided, ``auto_io=False``): run an
      arbitrary Python script as-is. No CLI args are injected — the script
      handles all I/O itself (reading from ``/var/inputs/``, writing to
      ``/var/outputs/<name>`` manually).
    - **Command mode** (``command`` provided): run any shell command directly,
      e.g. a compiled binary or a shell pipeline.

    Call ``.run()`` on the returned sandbox to build the image and execute.

    Example — auto-IO mode (default, no boilerplate)::

        sandbox = flyte.sandbox.create(
            name="double",
            code="result = x * 2",
            inputs={"x": int},
            outputs={"result": int},
        )
        result = await sandbox.run.aio(x=21)  # returns 42

    Example — verbatim mode (complete Python script, full control)::

        sandbox = flyte.sandbox.create(
            name="etl",
            code=\"\"\"
                import json, pathlib
                data = json.loads(pathlib.Path("/var/inputs/payload").read_text())
                pathlib.Path("/var/outputs/total").write_text(str(sum(data["values"])))
            \"\"\",
            inputs={"payload": File},
            outputs={"total": int},
            auto_io=False,
        )

    Example — command mode::

        sandbox = flyte.sandbox.create(
            name="test-runner",
            command=["/bin/bash", "-c", pytest_cmd],
            arguments=["_", "/var/inputs/solution.py", "/var/inputs/tests.py"],
            inputs={"solution.py": File, "tests.py": File},
            outputs={"exit_code": str},
        )

    Args:
        name: Sandbox name. Derives task and image names.
        code: Python source to run (auto-IO or verbatim mode). Mutually
            exclusive with ``command``.
        inputs: Input type declarations. Supported types:

            - Primitive: ``int``, ``float``, ``str``, ``bool``
            - Date/time: ``datetime.datetime``, ``datetime.timedelta``
            - IO handles: ``flyte.io.File``, ``flyte.io.Dir``
              (bind-mounted at ``/var/inputs/<name>``; available as a path
              string in auto-IO mode)

        outputs: Output type declarations. Supported types:

            - Primitive: ``int``, ``float``, ``str``, ``bool``
            - Date/time: ``datetime.datetime`` (ISO-8601), ``datetime.timedelta``
            - IO handles: ``flyte.io.File``, ``flyte.io.Dir``
              (user code must write the file/dir to ``/var/outputs/<name>``)

        command: Entrypoint command (command mode). Mutually exclusive with ``code``.
        arguments: Arguments forwarded to ``command`` (command mode only).
        packages: Python packages to install via pip.
        system_packages: System packages to install via apt.
        additional_commands: Extra Dockerfile ``RUN`` commands.
        resources: CPU / memory resources for the container.
        image_config: Registry and Python version settings.
        image_name: Explicit image name, overrides the auto-generated one.
        image: Pre-built image URI. Skips the build step if provided.
        auto_io: When ``True`` (default), Flyte wraps ``code`` with an
            auto-generated argparse preamble and output-writing epilogue so
            declared inputs are available as local variables and scalar outputs
            are collected automatically — no boilerplate needed. When
            ``False``, ``code`` is run verbatim and must handle all I/O itself.
        block_network: Block all outbound network access inside the container.
            Defaults to ``True`` (``--network none``).
        retries: Number of task retries on failure.
        timeout: Task timeout in seconds.
        env_vars: Environment variables available inside the container.
        secrets: Flyte :class:`~flyte.Secret` objects to mount.
        cache: Cache behaviour — ``"auto"``, ``"override"``, or ``"disable"``.

    Returns:
        Configured sandbox ready to ``.run()``.
    """
    if code is not None and command is not None:
        raise ValueError("'code' and 'command' are mutually exclusive.")

    _supported_names = ", ".join(
        sorted(f"datetime.{t.__name__}" if t.__module__ == "datetime" else t.__name__ for t in _SUPPORTED_TYPES)
    )
    for label, mapping in (("input", inputs or {}), ("output", outputs or {})):
        for param_name, typ in mapping.items():
            if typ not in _SUPPORTED_TYPES:
                raise TypeError(
                    f"Unsupported {label} type for '{param_name}': {typ!r}. Supported types: {_supported_names}."
                )

    return _Sandbox(
        name=name,
        code=code,
        inputs=inputs or {},
        outputs=outputs or {},
        command=command,
        arguments=arguments,
        packages=packages or [],
        system_packages=system_packages or [],
        additional_commands=additional_commands or [],
        resources=resources,
        image_config=image_config,
        image_name=image_name,
        image=image,
        auto_io=auto_io,
        block_network=block_network,
        retries=retries,
        timeout=timeout,
        env_vars=env_vars,
        secrets=secrets,
        cache=cache,
    )
