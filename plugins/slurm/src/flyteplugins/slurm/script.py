"""Rendering of sbatch scripts for Flyte tasks.

Everything in this module is a pure function of its inputs so the generated
script can be unit-tested without a Slurm cluster.
"""

from __future__ import annotations

import re
import shlex
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_SBATCH_KEY_RE = re.compile(r"^[a-z][a-z0-9-]*$")

# Enroot does not carry the image's PATH into the container -- it resets PATH from its own
# environ.d -- so a bare entrypoint name like `a0` is not found, even though the image puts
# it on PATH and the same task works as a Kubernetes pod. VIRTUAL_ENV *is* propagated, so
# re-derive the venv's bin directory inside the container before exec'ing the entrypoint.
# `exec "$@"` hands over argv untouched, so no argument needs re-quoting.
_PATH_SHIM = 'export PATH="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin:}$PATH"; exec "$@"'

# Well-known sbatch options that the plugin config exposes as first-class fields.
_FIELD_TO_SBATCH = {
    "partition": "partition",
    "nodes": "nodes",
    "ntasks": "ntasks",
    "cpus_per_task": "cpus-per-task",
    "gres": "gres",
    "gpus_per_node": "gpus-per-node",
    "mem": "mem",
    "time_limit": "time",
    "account": "account",
    "qos": "qos",
    "reservation": "reservation",
    "constraint": "constraint",
}


def pyxis_image_ref(image: str) -> str:
    """Convert an OCI image reference into the form Pyxis/Enroot expects.

    Enroot addresses registries with `REGISTRY#IMAGE:TAG` rather than the
    `REGISTRY/IMAGE:TAG` form used by Docker. Local squashfs paths and
    references that already use `#` are returned unchanged.

    >>> pyxis_image_ref("ghcr.io/flyteorg/flyte:py3.12-v2.0.0")
    'ghcr.io#flyteorg/flyte:py3.12-v2.0.0'
    >>> pyxis_image_ref("python:3.12-slim")
    'python:3.12-slim'
    >>> pyxis_image_ref("/jail/images/train.sqsh")
    '/jail/images/train.sqsh'
    """
    if image.startswith(("/", "./")) or "#" in image:
        return image
    image = image.removeprefix("docker://")
    first, sep, rest = image.partition("/")
    # A registry host contains a dot or a port, or is "localhost". Anything
    # else is a Docker Hub namespace and needs no rewriting.
    if sep and ("." in first or ":" in first or first == "localhost"):
        return f"{first}#{rest}"
    return image


def _sbatch_directive(key: str, value: object) -> str:
    if not _SBATCH_KEY_RE.match(key):
        raise ValueError(f"Invalid sbatch option name: {key!r}")
    if value is True or value is None:
        return f"#SBATCH --{key}"
    rendered = str(value)
    # A newline would end the `#SBATCH` comment and turn everything after it into script
    # body, which Slurm runs as the SSH user. Values reach here from task config
    # (`sbatch_options`, `working_dir`) so they are not necessarily trusted.
    if "\n" in rendered or "\r" in rendered:
        raise ValueError(f"sbatch option {key!r} may not contain a newline: {rendered!r}")
    return f"#SBATCH --{key}={rendered}"


def sbatch_directives(
    job_name: str,
    stdout_path: str,
    stderr_path: str,
    fields: Mapping[str, object],
    extra: Optional[Mapping[str, object]] = None,
) -> List[str]:
    """Build the `#SBATCH` header lines.

    `fields` are the first-class config fields (partition, nodes, ...),
    `extra` is the raw passthrough map. Passthrough wins on conflict so a
    site-specific flag can always override a first-class default.
    """
    options: Dict[str, object] = {
        "job-name": job_name,
        "output": stdout_path,
        "error": stderr_path,
    }
    for field, flag in _FIELD_TO_SBATCH.items():
        value = fields.get(field)
        if value is not None:
            options[flag] = value
    if extra:
        for key, value in extra.items():
            options[str(key)] = value
    return [_sbatch_directive(k, v) for k, v in options.items()]


def _export_lines(env: Mapping[str, str]) -> List[str]:
    lines = []
    for key in sorted(env):
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
        lines.append(f"export {key}={shlex.quote(str(env[key]))}")
    return lines


def render_container_job(
    *,
    job_name: str,
    stdout_path: str,
    stderr_path: str,
    image: str,
    command: Sequence[str],
    env: Mapping[str, str],
    sbatch_fields: Mapping[str, object],
    sbatch_extra: Optional[Mapping[str, object]] = None,
    container_mounts: Iterable[str] = (),
    container_workdir: Optional[str] = None,
    srun_extra_args: Sequence[str] = (),
) -> str:
    """Render an sbatch script that runs `command` inside `image` via Pyxis.

    This is the native-task path: `command` is the task's own Flyte
    entrypoint, so the job behaves exactly like the equivalent Kubernetes pod.
    """
    if not command:
        raise ValueError("A container job needs a non-empty command")

    srun = ["srun", f"--container-image={pyxis_image_ref(image)}"]
    mounts = ",".join(container_mounts)
    if mounts:
        srun.append(f"--container-mounts={mounts}")
    if container_workdir:
        srun.append(f"--container-workdir={container_workdir}")
    srun.extend(srun_extra_args)
    srun.extend(["bash", "-c", _PATH_SHIM, "--", *command])

    lines = [
        "#!/bin/bash",
        *sbatch_directives(job_name, stdout_path, stderr_path, sbatch_fields, sbatch_extra),
        "",
        "set -euo pipefail",
        *_export_lines(env),
        "",
        shlex.join(srun),
        "",
    ]
    return "\n".join(lines)


def split_leading_directives(script: str) -> Tuple[List[str], str]:
    """Split a user script into its leading `#SBATCH` block and the rest.

    `sbatch` only reads directives that appear before the first executable line, so this
    collects exactly the ones Slurm would have honoured had the script been submitted
    directly. A `#SBATCH` line further down is already inert; it stays in the body rather
    than being promoted, so wrapping a script cannot start applying an option the cluster
    was ignoring.

    The shebang is dropped -- a script may only have one and it must be the first line.
    """
    lines = script.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]

    directives: List[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#SBATCH"):
            directives.append(stripped)
            continue
        if not stripped or stripped.startswith("#"):
            continue  # blank lines and ordinary comments do not end the directive block
        return directives, "\n".join(lines[index:])
    return directives, ""


def render_script_job(
    *,
    job_name: str,
    stdout_path: str,
    stderr_path: str,
    script: str,
    env: Mapping[str, str],
    sbatch_fields: Mapping[str, object],
    sbatch_extra: Optional[Mapping[str, object]] = None,
) -> str:
    """Wrap a user-supplied batch script so it can be submitted by Flyte.

    The script's own leading `#SBATCH` directives are preserved and emitted first; ours
    follow, so on a duplicated option ours wins -- `sbatch` applies options in order and
    the last occurrence takes effect. Both blocks sit above the `export` lines, because
    `sbatch` stops reading directives at the first executable line: emitting the exports
    in between would silently drop every directive the script carries.
    """
    user_directives, body = split_leading_directives(script)
    if body and not body.endswith("\n"):
        body += "\n"
    lines = [
        "#!/bin/bash",
        *user_directives,
        *sbatch_directives(job_name, stdout_path, stderr_path, sbatch_fields, sbatch_extra),
        "",
        *_export_lines(env),
        "",
        body,
    ]
    return "\n".join(lines)
