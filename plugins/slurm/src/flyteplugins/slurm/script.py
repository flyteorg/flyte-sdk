"""Rendering of sbatch scripts for Flyte tasks.

Everything in this module is a pure function of its inputs so the generated
script can be unit-tested without a Slurm cluster.
"""

from __future__ import annotations

import re
import shlex
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

_SBATCH_KEY_RE = re.compile(r"^[a-z][a-z0-9-]*$")

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

    Enroot addresses registries with ``REGISTRY#IMAGE:TAG`` rather than the
    ``REGISTRY/IMAGE:TAG`` form used by Docker. Local squashfs paths and
    references that already use ``#`` are returned unchanged.

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
    return f"#SBATCH --{key}={value}"


def sbatch_directives(
    job_name: str,
    stdout_path: str,
    stderr_path: str,
    fields: Mapping[str, object],
    extra: Optional[Mapping[str, object]] = None,
) -> List[str]:
    """Build the ``#SBATCH`` header lines.

    ``fields`` are the first-class config fields (partition, nodes, ...),
    ``extra`` is the raw passthrough map. Passthrough wins on conflict so a
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
    """Render an sbatch script that runs ``command`` inside ``image`` via Pyxis.

    This is the native-task path: ``command`` is the task's own Flyte
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
    srun.extend(command)

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

    The user's script is embedded verbatim after our directives and exports.
    If it carries its own shebang and ``#SBATCH`` lines, Slurm honours ours
    (they come first) and treats theirs as comments, so a script that already
    works on the cluster keeps working unchanged.
    """
    body = script if script.endswith("\n") else script + "\n"
    if body.startswith("#!"):
        # Drop the user's shebang: a script can only have one, and it has to be the first line.
        body = body.split("\n", 1)[1] if "\n" in body else ""
    lines = [
        "#!/bin/bash",
        *sbatch_directives(job_name, stdout_path, stderr_path, sbatch_fields, sbatch_extra),
        "",
        *_export_lines(env),
        "",
        body,
    ]
    return "\n".join(lines)
