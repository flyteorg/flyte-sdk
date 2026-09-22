"""Rendering of sbatch scripts for Flyte tasks.

Everything in this module is a pure function of its inputs so the generated
script can be unit-tested without a Slurm cluster.
"""

from __future__ import annotations

import re
import shlex
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_SBATCH_KEY_RE = re.compile(r"^[a-z][a-z0-9-]*$")

#: Container runtimes a native `slurm` task can be launched with. Pyxis is the default
#: because it is what NVIDIA-shaped GPU clusters ship; Apptainer covers the traditional
#: HPC sites where Pyxis is not installed.
CONTAINER_RUNTIMES = ("pyxis", "apptainer")

# Options the plugin owns and a task may not set through `sbatch_options`. `output` and
# `error` are the dangerous pair: a task that prints an SSH public key and redirects
# output to the submitting user's `~/.ssh/authorized_keys` would get a shell as that
# shared account. The rest either break the connector's bookkeeping (`job-name`) or
# change what runs and as whom.
_RESERVED_SBATCH_OPTIONS = frozenset({"job-name", "output", "error", "chdir", "wrap", "uid", "gid"})

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


# Image reference schemes Apptainer understands as-is. Anything else is assumed to be a
# registry reference and gets the `docker://` prefix.
_APPTAINER_SCHEMES = ("docker://", "oras://", "library://", "shub://", "docker-daemon://")


def apptainer_image_ref(image: str) -> str:
    """Convert an OCI image reference into the form Apptainer expects.

    Apptainer addresses registries with a URI scheme rather than Enroot's `#` form, and
    takes a local `.sif` by path.

    >>> apptainer_image_ref("ghcr.io/flyteorg/flyte:py3.12-v2.0.0")
    'docker://ghcr.io/flyteorg/flyte:py3.12-v2.0.0'
    >>> apptainer_image_ref("/jail/images/train.sif")
    '/jail/images/train.sif'
    >>> apptainer_image_ref("docker://python:3.12-slim")
    'docker://python:3.12-slim'
    """
    if image.startswith(("/", "./")) or image.startswith(_APPTAINER_SCHEMES):
        return image
    return f"docker://{image}"


def _container_invocation(
    runtime: str,
    image: str,
    mounts: Sequence[str],
    workdir: Optional[str],
) -> List[str]:
    """Build the runtime-specific part of the srun line.

    Pyxis takes flags on `srun` itself; Apptainer is an ordinary command that wraps the
    payload. Everything else about the job -- directives, exports, the entrypoint -- is
    identical, which is why this is the only place the runtime matters.
    """
    if runtime == "pyxis":
        parts = [f"--container-image={pyxis_image_ref(image)}"]
        if mounts:
            parts.append(f"--container-mounts={','.join(mounts)}")
        if workdir:
            parts.append(f"--container-workdir={workdir}")
        return parts

    if runtime == "apptainer":
        parts = ["apptainer", "exec"]
        if mounts:
            parts.extend(["--bind", ",".join(mounts)])
        if workdir:
            parts.extend(["--pwd", workdir])
        parts.append(apptainer_image_ref(image))
        return parts

    raise ValueError(f"Unknown container_runtime {runtime!r}; expected one of: {', '.join(CONTAINER_RUNTIMES)}.")


def _sbatch_directive(key: str, value: object) -> str:
    if not _SBATCH_KEY_RE.match(key):
        raise ValueError(f"Invalid sbatch option name: {key!r}")
    if value is True or value is None:
        return f"#SBATCH --{key}"
    rendered = str(value)
    # Any whitespace is rejected, not just newlines. A newline ends the `#SBATCH` comment
    # and turns the rest into script body that Slurm runs as the SSH user; a space lets a
    # single value smuggle in a second option. Values reach here from task config
    # (`sbatch_options`, `working_dir`), so they are not necessarily trusted.
    if rendered != rendered.strip() or any(c.isspace() for c in rendered):
        raise ValueError(
            f"sbatch option {key!r} may not contain whitespace: {rendered!r}. "
            "Whitespace is refused because a single value could otherwise smuggle in a "
            "second option, or end the directive entirely."
        )
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
            name = str(key)
            if name in _RESERVED_SBATCH_OPTIONS:
                raise ValueError(
                    f"sbatch option {name!r} is set by the plugin and cannot be overridden from "
                    f"task config. Reserved: {', '.join(sorted(_RESERVED_SBATCH_OPTIONS))}."
                )
            options[name] = value
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
    container_runtime: str = "pyxis",
) -> str:
    """Render an sbatch script that runs `command` inside `image` via Pyxis.

    This is the native-task path: `command` is the task's own Flyte
    entrypoint, so the job behaves exactly like the equivalent Kubernetes pod.
    """
    if not command:
        raise ValueError("A container job needs a non-empty command")

    # Pin the task to a single process. Without this, `nodes=2` and no `ntasks` gives srun
    # its default of one task per node, so the Flyte entrypoint starts once per node and
    # every copy writes the same output prefix. A native task is single-process by design;
    # multi-node work belongs in a `slurm_script` task that drives srun itself.
    srun = ["srun", "--nodes=1", "--ntasks=1"]
    # srun's own extras come before the runtime, so they apply to the step rather than
    # being handed to the container command.
    srun.extend(srun_extra_args)
    srun.extend(_container_invocation(container_runtime, image, list(container_mounts), container_workdir))
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
