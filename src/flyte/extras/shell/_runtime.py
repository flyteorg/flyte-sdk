from __future__ import annotations

import pathlib
import shlex
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type, Union

import flyte
from flyte.extras._container import ContainerTask
from flyte.io import Dir, File

from ._render import _DICT_SEP, _render_command
from ._types import (
    _SCALAR_TYPES,
    FlagSpec,
    Glob,
    Stderr,
    Stdout,
    _classify_input,
    _is_dict_str_str,
    _is_optional,
    _ProcessResult,
    _validate_outputs,
    listMode,
)


@dataclass
class _Shell:
    """Configured shell task. Returned by :func:`create`."""

    name: str
    image: Union[str, flyte.Image]
    inputs: dict[str, Type]
    outputs: dict[str, Any]
    script: str
    flag_aliases: dict[str, FlagSpec] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    shell: str = "/bin/bash"
    debug: bool = False
    input_data_dir: pathlib.Path = pathlib.Path("/var/inputs")
    output_data_dir: pathlib.Path = pathlib.Path("/var/outputs")
    resources: Optional[flyte.Resources] = None
    retries: int = 0
    timeout: Optional[int] = None
    cache: str = "auto"
    block_network: bool = False
    env_vars: Optional[dict[str, str]] = None
    secrets: Optional[list] = None
    local_logs: bool = True
    _task: Optional["_ShellContainerTask"] = field(default=None, repr=False, compare=False)
    _env: Optional["flyte.TaskEnvironment"] = field(default=None, repr=False, compare=False)
    _resolved_image_uri: Optional[str] = field(default=None, repr=False, compare=False)

    def _container_inputs(self) -> dict[str, Any]:
        wired: dict[str, Any] = {}
        for name, tp in self.inputs.items():
            is_opt, inner = _is_optional(tp)
            if _is_dict_str_str(inner):
                wired[name] = str
            elif is_opt and inner in _SCALAR_TYPES:
                wired[name] = str
            elif is_opt:
                wired[name] = Optional[inner]
            else:
                wired[name] = inner
        return wired

    def _container_outputs(self) -> dict[str, Type]:
        wired: dict[str, Type] = {}
        for name, spec in self.outputs.items():
            if isinstance(spec, Glob):
                wired[name] = Dir
            elif isinstance(spec, (Stdout, Stderr)):
                wired[name] = spec.type
            else:
                wired[name] = spec
        return wired

    def _build_command(self) -> list[str]:
        body, positional_templates = _render_command(
            script=self.script,
            inputs=self.inputs,
            outputs=self.outputs,
            flag_specs={name: FlagSpec.coerce(name, self.flag_aliases.get(name)) for name in self.inputs},
            input_data_dir=self.input_data_dir,
            output_data_dir=self.output_data_dir,
        )

        stdout_target = self.output_data_dir / "_stdout"
        stderr_target = self.output_data_dir / "_stderr"
        for name, spec in self.outputs.items():
            if isinstance(spec, Stdout):
                stdout_target = self.output_data_dir / name
            elif isinstance(spec, Stderr):
                stderr_target = self.output_data_dir / name

        mkdirs = [
            f"mkdir -p {shlex.quote(str(self.output_data_dir / name))}"
            for name, spec in self.outputs.items()
            if isinstance(spec, Glob) or (isinstance(spec, type) and issubclass(spec, Dir))
        ]
        mkdir_preamble = "; ".join(mkdirs) + ";" if mkdirs else ""

        debug_preamble = ""
        if self.debug:
            debug_preamble = f"echo \"--- shell task: rendered script ---\" >&2; cat <<'_EOF_' >&2\n{body}\n_EOF_\n"

        wrapped = (
            f"{mkdir_preamble} "
            "set -o pipefail; "
            f"( {debug_preamble}{body} ) > {stdout_target} 2> {stderr_target}; "
            "_rc=$?; "
            f"echo $_rc > {self.output_data_dir / '_returncode'}; "
            "exit $_rc"
        )

        return [self.shell, "-c", wrapped, "_shell_task", *positional_templates]

    def as_task(self) -> "_ShellContainerTask":
        if self._task is None:
            self._task = _ShellContainerTask(self)
        return self._task

    async def _resolve_image_uri(self) -> str:
        if self._resolved_image_uri is not None:
            return self._resolved_image_uri

        if isinstance(self.image, str):
            self._resolved_image_uri = self.image
            return self._resolved_image_uri

        result = await flyte.build.aio(self.image)
        if result.uri is None:
            raise RuntimeError(
                f"Image build for shell task {self.name!r} returned no URI. "
                f"If using the remote builder asynchronously, wait for the "
                f"build to finish before calling the task."
            )

        self._resolved_image_uri = result.uri
        return self._resolved_image_uri

    @property
    def env(self) -> "flyte.TaskEnvironment":
        if self._env is None:
            self._env = flyte.TaskEnvironment.from_task(self.name, self.as_task())
        return self._env

    async def __call__(self, **kwargs) -> Any:
        """Run the shell wrapper and restore user-facing output types.

        The underlying serialized task still exposes ``Glob`` outputs as
        ``Dir`` on the wire. This wrapper converts those back to
        ``list[File]`` before returning to Python callers.
        """
        uri = await self._resolve_image_uri()
        task = self.as_task()

        if not isinstance(task._image, flyte.Image) or task._image.uri != uri:
            task._image = flyte.Image.from_base(uri)

        raw = await task(**(await self._prepare_kwargs(kwargs)))
        return await self._unpack_outputs(raw)

    async def _unpack_outputs(self, raw: Any) -> Any:
        """Convert wire-typed outputs back to user-facing types.

        Currently only :class:`Glob` needs unpacking: the runtime/wire type
        is ``Dir``, while the Python-facing type is ``list[File]``.
        """
        single = not isinstance(raw, tuple)
        values: list[Any] = [raw] if single else list(raw)
        items = list(self.outputs.items())

        if len(values) != len(items):
            return raw

        unpacked: list[Any] = []
        for (_, spec), value in zip(items, values):
            if isinstance(spec, Glob) and isinstance(value, Dir):
                local = await value.download() if hasattr(value, "download") else value.path
                matched = sorted(pathlib.Path(str(local)).glob(spec.pattern))
                unpacked.append([await File.from_local(str(p)) for p in matched if p.is_file()])
            else:
                unpacked.append(value)
        return unpacked[0] if single else tuple(unpacked)

    async def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.defaults:
            kwargs = {**self.defaults, **kwargs}

        out: dict[str, Any] = {}
        for name, tp in self.inputs.items():
            is_opt, _ = _is_optional(tp)
            kind = _classify_input(name, tp)

            if name not in kwargs:
                if not is_opt:
                    raise TypeError(f"Missing required input: {name!r}")
                if kind in ("scalar", "bool", "dict_str"):
                    out[name] = ""
                else:
                    out[name] = None
                continue

            value = kwargs[name]

            if value is None and is_opt:
                if kind in ("scalar", "bool", "dict_str"):
                    out[name] = ""
                else:
                    out[name] = None
                continue

            if kind == "dict_str" and isinstance(value, dict):
                parts: list[str] = []
                for k, v in value.items():
                    if _DICT_SEP in k or _DICT_SEP in v:
                        raise ValueError(
                            f"dict input {name!r}: keys/values cannot contain the record-separator byte (\\x1e)."
                        )
                    parts.append(k)
                    parts.append(v)
                out[name] = _DICT_SEP.join(parts)
                continue

            if is_opt and kind in ("scalar", "bool"):
                out[name] = "true" if (kind == "bool" and value) else "false" if kind == "bool" else str(value)
                continue

            out[name] = value
        return out


class _ShellContainerTask(ContainerTask):
    """ContainerTask subclass that overrides output collection for shell tasks."""

    def __init__(self, shell: _Shell):
        self._shell_spec = shell
        super().__init__(
            name=shell.name,
            image=shell.image,
            command=shell._build_command(),
            inputs=shell._container_inputs(),
            outputs=shell._container_outputs(),
            input_data_dir=str(shell.input_data_dir),
            output_data_dir=str(shell.output_data_dir),
            resources=shell.resources or flyte.Resources(cpu=1, memory="1Gi"),
            retries=shell.retries,
            timeout=shell.timeout,
            cache=shell.cache,
            block_network=shell.block_network,
            env_vars=shell.env_vars,
            secrets=shell.secrets,
            local_logs=shell.local_logs,
        )

    async def _get_output(self, output_directory: pathlib.Path) -> Tuple[Any, ...]:  # type: ignore[override]
        pr = await _read_process_result(output_directory)
        try:
            return await super()._get_output(output_directory)
        except (FileNotFoundError, ValueError, OSError) as e:
            raise FileNotFoundError(
                f"{e}\n\n"
                f"Script exited with returncode={pr.returncode}.\n"
                f"--- stdout ({len(pr.stdout)} bytes) ---\n"
                f"{_truncate(pr.stdout)}"
                f"--- stderr ({len(pr.stderr)} bytes) ---\n"
                f"{_truncate(pr.stderr)}"
            ) from e


def _validate_defaults(defaults: dict[str, Any], inputs: dict[str, Type]) -> dict[str, Any]:
    """Validate that every default key exists in ``inputs`` and that the
    value's Python type matches the declared input type.

    A ``None`` default is rejected — the same effect is achieved by
    declaring the input as ``T | None`` and omitting it from ``defaults``.
    """
    for name, value in defaults.items():
        if name not in inputs:
            raise KeyError(f"defaults references {name!r} which is not declared in inputs.")
        if value is None:
            raise ValueError(
                f"defaults[{name!r}] = None is redundant; declare the input as "
                f"`T | None` and omit it from defaults instead."
            )

        _, inner = _is_optional(inputs[name])
        kind = _classify_input(name, inputs[name])

        if kind == "file":
            if not isinstance(value, File):
                raise TypeError(f"defaults[{name!r}]: expected File, got {type(value).__name__}.")
        elif kind == "dir":
            if not isinstance(value, Dir):
                raise TypeError(f"defaults[{name!r}]: expected Dir, got {type(value).__name__}.")
        elif kind == "list_file":
            if not isinstance(value, list):
                raise TypeError(f"defaults[{name!r}]: expected list[File], got {type(value).__name__}.")
            if not all(isinstance(item, File) for item in value):
                raise TypeError(f"defaults[{name!r}]: list[File] requires every item to be a File.")
        elif kind == "bool":
            if not isinstance(value, bool):
                raise TypeError(f"defaults[{name!r}]: expected bool, got {type(value).__name__}.")
        elif kind == "dict_str":
            if not isinstance(value, dict):
                raise TypeError(f"defaults[{name!r}]: expected dict[str, str], got {type(value).__name__}.")
            for k, v in value.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise TypeError(f"defaults[{name!r}]: dict[str, str] requires string keys and values.")
        elif kind == "scalar":
            if inner is int:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"defaults[{name!r}]: expected int, got {type(value).__name__}.")
            elif inner is float:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"defaults[{name!r}]: expected float, got {type(value).__name__}.")
            elif inner is str:
                if not isinstance(value, str):
                    raise TypeError(f"defaults[{name!r}]: expected str, got {type(value).__name__}.")
            else:
                raise AssertionError(inner)
    return dict(defaults)


def _truncate(s: str, limit: int = 4000) -> str:
    if len(s) <= limit:
        return s if s.endswith("\n") else s + "\n"
    return s[:limit] + f"\n... [truncated {len(s) - limit} bytes]\n"


async def _read_process_result(output_directory: pathlib.Path) -> _ProcessResult:
    def read(p: pathlib.Path) -> str:
        try:
            return p.read_text()
        except FileNotFoundError:
            return ""

    rc_text = read(output_directory / "_returncode").strip()
    rc = int(rc_text) if rc_text else -1
    return _ProcessResult(
        returncode=rc,
        stdout=read(output_directory / "_stdout"),
        stderr=read(output_directory / "_stderr"),
    )


def create(
    name: str,
    *,
    image: Union[str, flyte.Image],
    inputs: Optional[dict[str, Type]] = None,
    outputs: Optional[dict[str, Any]] = None,
    script: str,
    flag_aliases: Optional[dict[str, Union[str, Tuple[str, listMode], FlagSpec]]] = None,
    defaults: Optional[dict[str, Any]] = None,
    shell: str = "/bin/bash",
    debug: bool = False,
    resources: Optional[flyte.Resources] = None,
    retries: int = 0,
    timeout: Optional[int] = None,
    cache: str = "auto",
    block_network: bool = False,
    env_vars: Optional[dict[str, str]] = None,
    secrets: Optional[list] = None,
    local_logs: bool = True,
) -> _Shell:
    """Wrap a CLI tool packaged in a container as a Flyte task.

    Args:
        name: Task name; should be unique within the project.
        image: Either a pre-built URI string
            (e.g. ``"quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_0"``,
            ``"debian:12-slim"``) or a :class:`flyte.Image` /
            ImageSpec instance (layered: base + apt / pip / Dockerfile
            layers). When you pass a ``flyte.Image``, the shell layer
            builds it for you on first call via :func:`flyte.build` —
            using the configured builder (``cfg.image_builder``:
            ``"local"`` by default, ``"remote"`` when opted in) — and
            hands the resulting URI down to ContainerTask. Subsequent
            calls reuse the cached URI; the build engine itself is also
            memoised, so cross-task duplication is cheap.

            .. important::
               **Requirements on the image:**

               1. ``bash`` (4+) at ``/bin/bash`` — the generated preamble
                  uses bash-only features (arrays, ``$'\\x1e'`` ANSI-C
                  quoting, ``read -ra``, ``$((..))`` arith, ``<<<``
                  here-strings, ``set -o pipefail``).
               2. **No custom ENTRYPOINT.** ContainerTask passes the bash
                  invocation via ``CMD``; if the image sets
                  ``ENTRYPOINT=["..."]``, docker prepends it and the
                  resulting invocation breaks.
        inputs: Mapping of input name to type. Supported types:

            - ``File``, ``Dir`` — mounted at ``/var/inputs/<name>``
            - ``list[File]`` — mounted under ``/var/inputs/<name>/`` and
              expanded as ``${name}/*`` (or via ``{flags.<name>}`` with a
              ``list_mode``)
            - ``dict[str, str]`` — passthrough "extras" dict; values are
              **strings only** (see recipes below)
            - ``int``, ``float``, ``str``, ``bool`` scalars
            - ``T | None`` of any of the above (``None`` collapses to empty)

            **Recipes for things that look like they need a richer dict but
            don't:**

            - *Bool as a CLI switch* (``--verbose``) — declare ``verbose: bool``
              as a separate input and use ``{flags.verbose}``.
            - *Bool as a value* (``REMOVE_DUPLICATES=true``) — already works
              with ``dict[str, str]``; the value is just the string ``"true"``.
            - *List of values under a repeated flag* (``-I a.bam -I b.bam``) —
              declare ``list[File]`` (or another typed list) with
              ``flag_aliases={"name": ("-I", "repeat")}``.
            - *List of strings, comma-joined* (``--exclude a,b,c``) — pass a
              pre-joined string yourself: ``extras={"--exclude": "a,b,c"}``.

            Resist the urge to extend ``dict[str, str]`` to mixed value types
            — declaring inputs individually gives you better type hints, IDE
            autocomplete, and clearer error messages.
        outputs: Mapping of output name to declaration. Each value is
            either a **bare type** (the common case) or a small **collector**
            class for behaviour the type system can't express:

            - ``File`` — single file at ``/var/outputs/<name>``
            - ``Dir`` — directory at ``/var/outputs/<name>`` (wrapper
              pre-creates it via ``mkdir -p``)
            - ``int`` / ``float`` / ``str`` / ``bool`` — primitive; the
              script writes the value as text to ``/var/outputs/<name>``
              and CoPilot casts to the declared type
            - :class:`Glob` (``pattern="*"``) — pattern-filtered
              ``list[File]``. The wrapper pre-creates the directory; the
              script writes files into it; the serialized task exposes
              that output as ``Dir`` on the wire, and the Python shell
              wrapper unpacks it back to ``list[File]`` post-execution.
            - :class:`Stdout` (``type=File`` by default) — the wrapper
              redirects the script's stdout straight to
              ``/var/outputs/<name>``. ``type`` can also be a primitive,
              in which case the captured text is cast.
            - :class:`Stderr` — symmetric for stderr.

            All declared outputs live at ``/var/outputs/<name>``; the
            user references them as ``{outputs.<name>}`` in the script
            (except :class:`Stdout` / :class:`Stderr`, which are managed
            by the wrapper).
        script: Bash script template. Reference inputs as ``{inputs.x}``,
            CLI flags as ``{flags.x}``, and outputs as ``{outputs.x}``
            (which renders to ``/var/outputs/<x>``). :class:`Stdout` /
            :class:`Stderr` outputs cannot be referenced — the wrapper
            redirects the corresponding stream there for you.

            **Do not wrap ``{inputs.x}`` in your own quotes**. Scalar values
            travel through bash positional args (``$1``, ``$2``) so they
            survive arbitrary content (single quotes, tabs, dollar signs)
            without escaping, and the wrapper already emits scalar
            references as ``"${_VAL_name}"`` (quoted, single token). Wrapping
            them in ``"..."`` again breaks out of the wrapper's quoting and
            re-enables word splitting.
        flag_aliases: Per-input override for ``{flags.<name>}`` rendering.
            Values may be a string (just the flag, default join mode) or
            ``(flag, list_mode)`` to pick a list rendering mode (``"join"``,
            ``"repeat"``, ``"comma"``) or ``(flag, dict_mode)``
            for dicts (``"pairs"``, ``"equals"``).
        defaults: Per-input fallback value used when the caller omits that
            input at call time. Lets you mark inputs as "optional at call
            site" while still emitting their flag, independent of the
            ``T | None`` axis. The interaction with ``T | None`` is:

            ====================  =========================  =================================
            Type                  In ``defaults``            Behavior when caller omits
            ====================  =========================  =================================
            ``T``                 No                         ``TypeError`` at submit time
            ``T``                 Yes                        Default used; flag emitted
            ``T | None``          No                         Empty value; flag suppressed
            ``T | None``          Yes                        Default used; flag emitted
            ====================  =========================  =================================
        shell: Shell binary to use. Defaults to ``/bin/bash``.
        debug: If True, container prints the rendered script to stderr
            before running. Invaluable when authoring a new wrapper.
        resources, retries, timeout, cache, block_network, env_vars,
            secrets: Standard task knobs forwarded to ContainerTask.
        local_logs: When ``True`` (default), the rendered command and the
            container's captured stdout/stderr are emitted through the
            flyte logger at ``DEBUG`` level during local docker execution.
            Set to ``False`` to silence them entirely (e.g. when running
            many sub-tasks locally and per-task chatter would clutter
            output even at DEBUG). Only affects local docker execution;
            remote execution never invokes the code path that produces
            these messages.

    Returns:
        A configured :class:`_Shell` instance. Call it like a coroutine for
        local execution; access ``.env`` to plug it into a pipeline's
        ``depends_on`` for deploy-time image building and registration.

    Example::

        # bedtools.py
        from flyte.io import File
        from flyte.extras.shell import create, Glob

        bedtools_intersect = create(
            name="bedtools_intersect",
            image="quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_0",
            inputs={"a": File, "b": list[File], "wa": bool, "f": float},
            outputs={"bed": Glob("*.bed")},
            script=r'''
                bedtools intersect {flags.wa} \\
                    -a {inputs.a} \\
                    -b {inputs.b} \\
                    -f {inputs.f} \\
                    > {outputs.bed}/out.bed
            ''',
        )

        # user_pipeline.py
        import flyte
        from bio_modules.bedtools import bedtools_intersect

        env = flyte.TaskEnvironment(
            name="genomics_pipeline",
            depends_on=[bedtools_intersect.env],
        )

        @env.task
        async def pipeline(a: File, b: list[File]) -> list[File]:
            return await bedtools_intersect(a=a, b=b, wa=True, f=0.5)
    """
    inputs = inputs or {}
    outputs = outputs or {}

    # Sanity-check the generated helper names used by the shell renderer.
    generated_helpers: dict[str, str] = {}
    for n in inputs:
        for helper in (f"_VAL_{n}", f"_FLAG_{n}", f"_ARR_{n}"):
            if helper in generated_helpers:
                raise ValueError(
                    f"Input names {generated_helpers[helper]!r} and {n!r} collide in generated shell helper "
                    f"name {helper!r}. Rename one input."
                )
            generated_helpers[helper] = n

    for n, t in inputs.items():
        _classify_input(n, t)

    _validate_outputs(outputs)

    coerced_aliases: dict[str, FlagSpec] = {}
    for n, alias in (flag_aliases or {}).items():
        if n not in inputs:
            raise KeyError(f"flag_aliases references {n!r} which is not declared in inputs.")
        coerced_aliases[n] = FlagSpec.coerce(n, alias)

    validated_defaults: dict[str, Any] = _validate_defaults(defaults or {}, inputs)

    if not isinstance(image, (str, flyte.Image)):
        raise TypeError(f"image must be a URI string or a flyte.Image, got {type(image).__name__}.")

    return _Shell(
        name=name,
        image=image,
        inputs=dict(inputs),
        outputs=dict(outputs),
        script=script,
        flag_aliases=coerced_aliases,
        defaults=validated_defaults,
        shell=shell,
        debug=debug,
        resources=resources,
        retries=retries,
        timeout=timeout,
        cache=cache,
        block_network=block_network,
        env_vars=env_vars,
        secrets=secrets,
        local_logs=local_logs,
    )
