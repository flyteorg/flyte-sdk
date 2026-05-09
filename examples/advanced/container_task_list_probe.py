"""
Demonstrates `cli_container_task` from cli_container_task_helper.py.

This file used to be a hand-rolled probe that investigated how Flyte CoPilot
localizes list[File] / dict[str, File] inputs and composes argv via shell jq.
The findings of that probe are now baked into `cli_container_task`, so this
example just exercises the helper end-to-end:

  - list[str] flags are passed through verbatim
  - dict[str, str] kwargs become "<key> <value>" pairs (NUL-safe, so values
    can contain tabs/spaces — e.g. SAM read groups)
  - list[File] inputs are localized under /var/inputs/<name>/ and globbed in
    deterministic sorted order

dict[str, File] is intentionally not exercised — the helper doesn't expose it
because there's no canonical bio-CLI mapping (positional? --key value? sorted?).
If a real workflow needs it, declare each file as its own input.
"""

import tempfile

import flyte
from examples.advanced.cli_container_task_helper import cli_container_task
from flyte.io import File

probe = cli_container_task(
    name="probe_inputs_json",
    image=flyte.Image.from_debian_base().with_apt_packages("jq"),
    # The "tool" here is just `echo`, so the captured stdout shows the
    # composed argv that the helper would have handed to a real bio CLI.
    tool=["echo", "ARGV:"],
    flag_inputs={"flags": list[str]},
    kwarg_inputs={"extras": dict[str, str]},
    file_inputs={"ref": File, "reads": list[File]},
    outputs={"composed": File},
    stdout_to="composed",
)

env = flyte.TaskEnvironment(
    name="probe_env",
    depends_on=[flyte.TaskEnvironment.from_task("probe_inner", probe)],
)


@env.task
async def main() -> File:
    paths = []
    for i, body in enumerate(["hello from file 0\n", "hello from file 1\n"]):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{i}.txt", delete=False
        ) as f:
            f.write(body)
            paths.append(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix="_ref.txt", delete=False) as f:
        f.write("reference\n")
        ref_path = f.name

    return await probe(
        flags=["--threads", "8", "-v"],
        # Tab in the value proves NUL-delimited handling preserves whitespace.
        extras={"--memory": "4G", "--out": "/tmp/out", "-R": "@RG\tID:x"},
        ref=await File.from_local(ref_path),
        reads=[await File.from_local(p) for p in paths],
    )


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.with_runcontext(mode="remote").run(main)
    print(r.url)
