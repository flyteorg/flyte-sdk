"""Parse PEP 723 inline metadata out of pixi scripts and render it as a pixi manifest.

Pixi runs standalone Python scripts whose dependencies are declared in a PEP 723 block,
with pixi-specific settings under `[tool.pixi.*]`:

    # /// script
    # requires-python = ">=3.11"
    # dependencies = ["httpx"]
    #
    # [tool.pixi.workspace]
    # channels = ["conda-forge"]
    #
    # [tool.pixi.dependencies]
    # gdal = "*"
    # ///

See https://pixi.prefix.dev/latest/python/scripts/.

There is no `pixi install --script`: the only commands that materialize a script's
environment are `pixi run --script` (which runs the script) and they place it in a
content-addressed rattler cache directory. Neither is usable at image build time, so a
pixi script is lowered into an equivalent `pixi.toml` workspace manifest and installed
with the ordinary `pixi install --manifest-path` path instead.
"""

import pathlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import toml

# PEP 723 block: `# /// script` ... `# ///`, where every line in between is either `#`
# alone or starts with `# `.
_PEP723_BLOCK = re.compile(
    r"(?m)^# /// script\s*$\s(?P<content>(?:^#(?: .*)?$\s)*?)^# ///\s*$",
)

# Conda subdir for each container architecture flyte can build for.
_PLATFORM_TO_SUBDIR = {
    "linux/amd64": "linux-64",
    "linux/arm64": "linux-aarch64",
}

DEFAULT_CHANNELS: Tuple[str, ...] = ("conda-forge",)


@dataclass(frozen=True)
class PixiScriptMetadata:
    """The parsed PEP 723 block of a pixi script."""

    #: `requires-python`, the PEP 723 python constraint (e.g. ">=3.11").
    requires_python: Optional[str] = None
    #: `dependencies`, the PEP 723 PyPI requirements, as PEP 508 strings.
    dependencies: List[str] = field(default_factory=list)
    #: The whole `[tool.pixi]` table, verbatim (workspace, dependencies, pypi-dependencies,
    #: target, feature, activation, ...). Passed through to the generated manifest so that
    #: pixi settings flyte does not model still reach pixi.
    pixi: Dict[str, Any] = field(default_factory=dict)


def _extract_pep723_block(text: str) -> Optional[str]:
    """Return the TOML source of the script's PEP 723 block, or None if there isn't one."""
    match = _PEP723_BLOCK.search(text)
    if not match:
        return None
    lines = []
    for line in match.group("content").splitlines():
        # Each line is `#` alone or `# <content>`; strip exactly that prefix so that TOML
        # indentation inside the block is preserved.
        lines.append(line[2:] if line.startswith("# ") else line[1:])
    return "\n".join(lines)


def parse_pixi_script_file(path: pathlib.Path) -> PixiScriptMetadata:
    """Parse the PEP 723 block of the pixi script at `path`."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    # Parsing is cached because the image hash re-reads the script repeatedly, but the
    # cache is keyed on the file's mtime and size as well as its path: an edited script
    # must produce new metadata, or a dependency change would not rebuild the image.
    stat = path.stat()
    return _parse_pixi_script_file(path, stat.st_mtime_ns, stat.st_size)


@lru_cache
def _parse_pixi_script_file(path: pathlib.Path, mtime_ns: int, size: int) -> PixiScriptMetadata:
    raw_header = _extract_pep723_block(path.read_text(encoding="utf-8"))
    if raw_header is None:
        raise ValueError(
            f"No PEP 723 script metadata block found in {path}. A pixi script must declare its "
            "dependencies in a `# /// script` ... `# ///` block. "
            "See https://pixi.prefix.dev/latest/python/scripts/"
        )

    try:
        data = toml.loads(raw_header)
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML in the script metadata block of {path}: {e}")

    return PixiScriptMetadata(
        requires_python=data.get("requires-python"),
        dependencies=list(data.get("dependencies", [])),
        pixi=dict(data.get("tool", {}).get("pixi", {})),
    )


def platforms_for(platform: Tuple[str, ...]) -> Tuple[str, ...]:
    """Map container platforms (`linux/amd64`) onto conda subdirs (`linux-64`)."""
    try:
        return tuple(_PLATFORM_TO_SUBDIR[p] for p in platform)
    except KeyError as e:
        raise ValueError(
            f"Cannot build a pixi environment for platform {e.args[0]!r}: "
            f"supported platforms are {sorted(_PLATFORM_TO_SUBDIR)}."
        )


def _pypi_requirement_to_pixi(requirement: str) -> Tuple[str, Any]:
    """Convert one PEP 508 requirement string into a pixi `[pypi-dependencies]` entry."""
    from packaging.requirements import InvalidRequirement, Requirement

    try:
        req = Requirement(requirement)
    except InvalidRequirement as e:
        raise ValueError(f"Invalid PEP 508 requirement {requirement!r} in script dependencies: {e}")

    if req.marker is not None:
        # pixi expresses conditional dependencies with `[tool.pixi.target.<platform>]`
        # tables rather than with environment markers, so there is nothing faithful to
        # translate a marker into.
        raise ValueError(
            f"Environment markers are not supported in pixi script dependencies: {requirement!r}. "
            "Express platform-conditional dependencies with a "
            "`[tool.pixi.target.<platform>.pypi-dependencies]` table instead."
        )

    spec: Dict[str, Any] = {}
    if req.url:
        if req.url.startswith("git+"):
            url, _, rev = req.url[len("git+") :].partition("@")
            spec["git"] = url
            if rev:
                spec["rev"] = rev
        else:
            spec["url"] = req.url
    else:
        spec["version"] = str(req.specifier) if req.specifier else "*"

    if req.extras:
        spec["extras"] = sorted(req.extras)

    # Collapse the common `{version = "..."}` case to pixi's shorthand.
    if set(spec) == {"version"}:
        return req.name, spec["version"]
    return req.name, spec


def effective_platforms(metadata: PixiScriptMetadata, platforms: Tuple[str, ...]) -> Tuple[str, ...]:
    """The conda subdirs the generated workspace will support.

    A `platforms` key declared by the script under `[tool.pixi.workspace]` wins, since the
    script author stated it explicitly (and any lock file was resolved against it);
    otherwise the image's own platforms are used.
    """
    declared = metadata.pixi.get("workspace", {}).get("platforms")
    return tuple(declared) if declared else platforms


def check_platforms_supported(metadata: PixiScriptMetadata, platforms: Tuple[str, ...]) -> None:
    """Raise if the script's declared platforms do not cover every platform of the image.

    pixi refuses to install a workspace on a platform it does not list, so a script that
    declares fewer platforms than the image is built for fails partway through the build.
    Catching it here turns that into an actionable message.
    """
    declared = metadata.pixi.get("workspace", {}).get("platforms")
    if not declared:
        return
    missing = [p for p in platforms if p not in declared]
    if missing:
        raise ValueError(
            f"The pixi script declares platforms {list(declared)} under `[tool.pixi.workspace]`, "
            f"which do not cover {missing}, needed to build this image. Either add {missing} to the "
            f"script's `platforms`, or restrict the image with "
            f"`from_pixi_script(..., platform=(...,))`."
        )


def render_pixi_manifest(metadata: PixiScriptMetadata, platforms: Tuple[str, ...]) -> str:
    """Render `metadata` as the source of an equivalent `pixi.toml` workspace manifest.

    `platforms` are the conda subdirs the workspace should support, unless the script
    declares its own under `[tool.pixi.workspace]`.
    """
    workspace: Dict[str, Any] = dict(metadata.pixi.get("workspace", {}))
    # A workspace manifest, unlike a script, must state its channels and platforms: pixi
    # infers neither from the machine it is installing on.
    workspace.setdefault("channels", list(DEFAULT_CHANNELS))
    workspace["platforms"] = list(effective_platforms(metadata, platforms))

    dependencies: Dict[str, Any] = dict(metadata.pixi.get("dependencies", {}))
    if metadata.requires_python and "python" not in dependencies:
        # An explicit `[tool.pixi.dependencies].python` wins over `requires-python`.
        dependencies["python"] = metadata.requires_python

    pypi_dependencies: Dict[str, Any] = dict(metadata.pixi.get("pypi-dependencies", {}))
    for requirement in metadata.dependencies:
        name, spec = _pypi_requirement_to_pixi(requirement)
        # `[tool.pixi.pypi-dependencies]` is the more specific declaration, so it wins.
        pypi_dependencies.setdefault(name, spec)

    manifest: Dict[str, Any] = {"workspace": workspace}
    if dependencies:
        manifest["dependencies"] = dependencies
    if pypi_dependencies:
        manifest["pypi-dependencies"] = pypi_dependencies
    # Everything else pixi understands (target, feature, environments, activation, ...)
    # is passed through untouched.
    for key, value in metadata.pixi.items():
        if key not in ("workspace", "dependencies", "pypi-dependencies"):
            manifest[key] = value

    header = (
        "# Generated by flyte from the PEP 723 metadata of a pixi script.\n"
        "# Edit the script's `# /// script` block rather than this file.\n"
    )
    return header + toml.dumps(manifest)
