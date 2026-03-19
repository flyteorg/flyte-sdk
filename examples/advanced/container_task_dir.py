"""
ContainerTask with Dir input and Dir output.

This example shows how to pass a FlyteDirectory (Dir) into a raw container task and
produce a new Dir as output.  The container transforms each file in the input directory
and writes the results to the output directory.

Key rules for Dir in ContainerTask:
- Input:  reference via path (/var/inputs/<key>), NOT the {{.inputs.key}} template syntax.
- Output: container writes a sub-directory named after the output key inside output_data_dir.
          e.g. for outputs={"out_dir": Dir}, write files to /var/outputs/out_dir/
"""

import os
import tempfile

import flyte
from flyte.extras import ContainerTask
from flyte.io import Dir

# ── Step 1: create a sample Dir with a normal @task ──────────────────────────

creator_env = flyte.TaskEnvironment(name="dir_creator")


@creator_env.task
async def create_sample_dir(n: int = 3) -> Dir:
    """Create a temporary directory with n text files and upload it."""
    tmpdir = tempfile.mkdtemp()
    for i in range(n):
        with open(os.path.join(tmpdir, f"file{i}.txt"), "w") as f:
            f.write(f"line {i}\n")
    return await Dir.from_local(tmpdir)


# ── Step 2: ContainerTask that takes a Dir input and produces a Dir output ───
#
# Input rule:  Dir inputs land at /var/inputs/<key> — hardcode the path in the command.
# Output rule: write a sub-directory named after the output key inside /var/outputs/.

transform_dir_task = ContainerTask(
    name="transform_dir",
    image="alpine:3.18",
    input_data_dir="/var/inputs",
    output_data_dir="/var/outputs",
    inputs={"in_dir": Dir},
    outputs={"out_dir": Dir},
    # Read every *.txt from /var/inputs/in_dir, uppercase the content, write to out_dir.
    command=[
        "/bin/sh",
        "-c",
        "mkdir -p /var/outputs/out_dir && "
        "for f in /var/inputs/in_dir/*.txt; do "
        "  fname=$(basename $f); "
        "  tr '[:lower:]' '[:upper:]' < $f > /var/outputs/out_dir/$fname; "
        "done",
    ],
)

container_env = flyte.TaskEnvironment.from_task("container_dir_env", transform_dir_task)

env = flyte.TaskEnvironment(
    name="dir_example",
    depends_on=[creator_env, container_env],
)


# ── Step 3: orchestrating workflow ───────────────────────────────────────────


@env.task
async def main(n: int = 3) -> str:
    sample_dir = await create_sample_dir(n=n)

    out_dir = await transform_dir_task(in_dir=sample_dir)

    files = await out_dir.list_files()
    print(f"Output dir: {out_dir.path}")
    print(f"Files: {[f.name for f in files]}")
    return out_dir.path


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(main, n=3)
    print(run.url)