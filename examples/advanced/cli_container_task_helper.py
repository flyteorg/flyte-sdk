"""
Helper for the bio team: turn a typed spec of (flags, kwargs, files) into a
ContainerTask that composes argv for a CLI tool from inputs.json at runtime.

Bio CLIs (bwa, samtools, bcftools, gatk, ...) all want the same argv shape:

    tool <flag list...> <--key value pairs...> <positional files...>

Hand-rolling that shell with `jq | tr` is brittle. This helper lets the user
declare:

    bwa_mem = cli_container_task(
        name="bwa_mem",
        image=Image.from_debian_base().with_apt_packages("bwa", "jq"),
        tool=["bwa", "mem"],
        flag_inputs={"flags": list[str]},          # appended verbatim, e.g. ["-v", "-t", "8"]
        kwarg_inputs={"opts": dict[str, str]},     # rendered as "--key value", e.g. {"-R": "@RG..."}
        file_inputs={"ref": File, "reads": list[File]},  # positional, in declaration order
        outputs={"sam": File},
        stdout_to="sam",                            # capture stdout into /var/outputs/<name>
    )

The helper handles:
  - merging the typed inputs into one ContainerTask `inputs={}` schema
  - generating the /bin/sh preamble that uses jq to read /var/inputs/inputs.json
    and compose argv
  - expanding list[File] inputs that CoPilot localizes under /var/inputs/<key>/*
  - either exec'ing the tool or piping stdout to /var/outputs/<name>

Requires `jq` in the image.
"""

from __future__ import annotations

import shlex
from typing import Any

import flyte
from flyte.extras import ContainerTask
from flyte.io import File


def cli_container_task(
    *,
    name: str,
    image: flyte.Image | str,
    tool: list[str],
    flag_inputs: dict[str, type] | None = None,
    kwarg_inputs: dict[str, type] | None = None,
    file_inputs: dict[str, type] | None = None,
    outputs: dict[str, type] | None = None,
    stdout_to: str | None = None,
    extra_inputs: dict[str, type] | None = None,
    block_network: bool = False,
) -> ContainerTask:
    """
    Build a ContainerTask that composes argv for a CLI tool from typed inputs.

    Argv layout (in this order):
        tool[*] <flag_inputs in declaration order> <kwarg_inputs in declaration order> <file_inputs in declaration order>

    Parameters
    ----------
    name
        Task name.
    image
        Container image. MUST contain `jq` and the tool itself.
    tool
        Command + fixed leading args, e.g. ["bwa", "mem"] or ["samtools", "view"].
    flag_inputs
        Mapping of input-name to type. Each value must resolve to `list[str]` at
        runtime; elements are appended to argv verbatim.
    kwarg_inputs
        Mapping of input-name to type. Each value must resolve to `dict[str, str]`;
        each entry becomes two argv tokens: "<key> <value>".
    file_inputs
        Mapping of input-name to type. Each value must be `File` or `list[File]`.
        Single File becomes "/var/inputs/<name>". list[File] expands to every file
        under "/var/inputs/<name>/" (sorted by filename for determinism).
    outputs
        Output spec passed straight to ContainerTask.
    stdout_to
        If set, the tool's stdout is redirected to "/var/outputs/<stdout_to>"
        instead of being printed. The corresponding output must exist in `outputs`.
    extra_inputs
        Extra typed inputs passed to ContainerTask but NOT composed into argv —
        useful when the user needs an input for templating elsewhere or for
        side channels.
    block_network
        Forwarded to ContainerTask.
    """
    flag_inputs = flag_inputs or {}
    kwarg_inputs = kwarg_inputs or {}
    file_inputs = file_inputs or {}
    extra_inputs = extra_inputs or {}

    _check_no_collisions(flag_inputs, kwarg_inputs, file_inputs, extra_inputs)
    if stdout_to and (not outputs or stdout_to not in outputs):
        raise ValueError(f"stdout_to={stdout_to!r} must reference a key in outputs")

    inputs: dict[str, type] = {}
    inputs.update(flag_inputs)
    inputs.update(kwarg_inputs)
    inputs.update(file_inputs)
    inputs.update(extra_inputs)

    script = _build_shell(
        tool=tool,
        flag_keys=list(flag_inputs.keys()),
        kwarg_keys=list(kwarg_inputs.keys()),
        file_specs=[(k, _is_file_list(v)) for k, v in file_inputs.items()],
        stdout_to=stdout_to,
    )

    return ContainerTask(
        name=name,
        image=image,
        inputs=inputs,
        outputs=outputs,
        metadata_format="JSON",
        block_network=block_network,
        # bash (not sh) so we can use arrays + NUL-delimited reads, which is the
        # only way to preserve argv tokens that contain tabs/spaces (read groups,
        # paths with spaces, JSON-encoded options, ...).
        command=["/bin/bash", "-c", script],
    )


def _is_file_list(t: Any) -> bool:
    if t is File:
        return False
    origin = getattr(t, "__origin__", None)
    args = getattr(t, "__args__", ())
    if origin in (list,) and args and args[0] is File:
        return True
    raise TypeError(f"file_inputs values must be File or list[File], got {t!r}")


def _check_no_collisions(*maps: dict[str, type]) -> None:
    seen: set[str] = set()
    for m in maps:
        for k in m:
            if k in seen:
                raise ValueError(f"input name {k!r} appears in multiple input groups")
            seen.add(k)


def _build_shell(
    *,
    tool: list[str],
    flag_keys: list[str],
    kwarg_keys: list[str],
    file_specs: list[tuple[str, bool]],
    stdout_to: str | None,
) -> str:
    """Generate a bash script that composes argv from /var/inputs/inputs.json.

    Tokens are read NUL-delimited so values containing tabs/spaces (e.g. SAM
    read-group strings) survive intact.
    """
    lines: list[str] = [
        "set -euo pipefail",
        'INPUTS=/var/inputs/inputs.json',
        'command -v jq >/dev/null 2>&1 || { echo "cli_container_task: jq required in image" >&2; exit 2; }',
        '[ -f "$INPUTS" ] || { echo "cli_container_task: $INPUTS not found" >&2; exit 2; }',
        'ARGV=()',
    ]

    for k in flag_keys:
        sk = shlex.quote(k)
        lines.append(
            f'while IFS= read -r -d "" tok; do ARGV+=("$tok"); done '
            f'< <(jq -j --arg k {sk} \'.[$k][]? | tostring + "\\u0000"\' "$INPUTS")'
        )

    for k in kwarg_keys:
        sk = shlex.quote(k)
        lines.append(
            f'while IFS= read -r -d "" tok; do ARGV+=("$tok"); done '
            f'< <(jq -j --arg k {sk} '
            f'\'.[$k] | to_entries[]? | (.key + "\\u0000" + (.value|tostring) + "\\u0000")\' "$INPUTS")'
        )

    for k, is_list in file_specs:
        if is_list:
            # CoPilot localizes list[File] under /var/inputs/<k>/. Use a globbed
            # for-loop and sort for determinism. shopt nullglob keeps us silent
            # if the dir is empty (which would be a usage error anyway).
            lines.append('shopt -s nullglob')
            lines.append(
                f'for f in $(printf "%s\\n" /var/inputs/{k}/* | LC_ALL=C sort); do '
                f'ARGV+=("$f"); done'
            )
        else:
            lines.append(f'ARGV+=("/var/inputs/{k}")')

    tool_quoted = " ".join(shlex.quote(t) for t in tool)
    if stdout_to:
        lines.append('mkdir -p /var/outputs')
        lines.append(
            f'{tool_quoted} "${{ARGV[@]}}" > /var/outputs/{shlex.quote(stdout_to)}'
        )
    else:
        lines.append(f'{tool_quoted} "${{ARGV[@]}}"')

    return "\n".join(lines) + "\n"


# --- demo: echo stands in for a real bio CLI so this runs anywhere -----------

echo_demo = cli_container_task(
    name="echo_demo",
    image=flyte.Image.from_debian_base().with_apt_packages("jq"),
    tool=["echo", "demo:"],
    flag_inputs={"flags": list[str]},
    kwarg_inputs={"opts": dict[str, str]},
    file_inputs={"reads": list[File], "ref": File},
    outputs={"composed": File},
    stdout_to="composed",
)

env = flyte.TaskEnvironment(
    name="cli_helper_demo_env",
    depends_on=[flyte.TaskEnvironment.from_task("echo_demo_inner", echo_demo)],
)


@env.task
async def main() -> File:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix="_ref.fa", delete=False) as f:
        f.write(">ref\nACGT\n")
        ref_path = f.name
    read_paths = []
    for i, body in enumerate([">r0\nAAAA\n", ">r1\nTTTT\n"]):
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_r{i}.fa", delete=False) as f:
            f.write(body)
            read_paths.append(f.name)

    return await echo_demo(
        flags=["-v", "-t", "8"],
        opts={"--memory": "4G", "--out": "/tmp/out"},
        ref=await File.from_local(ref_path),
        reads=[await File.from_local(p) for p in read_paths],
    )


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.with_runcontext(mode="remote").run(main)
    print(r.url)
