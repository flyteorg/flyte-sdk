"""
Preserve input filenames in a raw ContainerTask with ``file_input_layout``.

By default (DIRECT) CoPilot stages a ``File`` input at an extensionless path
(``/var/inputs/<var>`` for a File, ``/var/inputs/<var>/<index>`` for list[File]
elements), which breaks tools that detect a file's format from its extension.

``file_input_layout="NAMED_DIR"`` instead stages each input inside a per-input
directory under its original basename, so names and extensions round-trip::

    fasta : File        -> /var/inputs/fasta/genome.fasta
    reads : list[File]  -> /var/inputs/reads/sample_R1.fastq.gz
                           /var/inputs/reads/sample_R2.fastq.gz

Consumers glob ``/var/inputs/<var>/*`` to read the input.

    flyte run examples/advanced/named_dir_file_input.py main        # single File
    flyte run examples/advanced/named_dir_file_input.py main_list   # list[File]
"""

from pathlib import Path

import flyte
from flyte.extras import ContainerTask
from flyte.io import File

# Single File: NAMED_DIR stages it at /var/inputs/fasta/<original-name>; the command
# globs the dir and reports whether the .fasta extension survived.
check_extension = ContainerTask(
    name="check_extension",
    image="alpine:3.20",
    inputs={"fasta": File},
    outputs={"name": str},
    input_data_dir="/var/inputs",
    output_data_dir="/var/outputs",
    file_input_layout="NAMED_DIR",
    command=[
        "/bin/sh",
        "-c",
        "set -eu; "
        "if [ -d /var/inputs/fasta ]; then f=$(ls /var/inputs/fasta/* | head -n1); else f=/var/inputs/fasta; fi; "
        'echo "staged input: $f"; '
        'case "$(basename "$f")" in '
        '  *.fasta) echo "RESULT: extension preserved (NAMED_DIR)" ;; '
        '  *) echo "RESULT: extension dropped (DIRECT)" ;; '
        "esac; "
        'printf \'%s\' "$(basename "$f")" | tee /var/outputs/name',
    ],
)

# list[File]: each element keeps its original basename under /var/inputs/reads/
# (e.g. sample_R1.fastq.gz) instead of the bare index 0/1.
check_reads = ContainerTask(
    name="check_reads",
    image="alpine:3.20",
    inputs={"reads": list[File]},
    outputs={"names": str},
    input_data_dir="/var/inputs",
    output_data_dir="/var/outputs",
    file_input_layout="NAMED_DIR",
    command=[
        "/bin/sh",
        "-c",
        "set -eu; "
        "ok=1; "
        "for f in /var/inputs/reads/*; do "
        '  echo "staged input: $f"; '
        '  case "$(basename "$f")" in '
        "    *.fastq.gz) ;; "
        "    *) ok=0 ;; "
        "  esac; "
        "done; "
        'if [ "$ok" -eq 1 ]; then echo "RESULT: all reads kept their extension (NAMED_DIR)"; '
        'else echo "RESULT: extensions dropped (DIRECT)"; fi; '
        "(cd /var/inputs/reads && echo *) | tee /var/outputs/names",
    ],
)

container_env = flyte.TaskEnvironment.from_task("named_dir_env", check_extension)
list_env = flyte.TaskEnvironment.from_task("named_dir_list_env", check_reads)
env = flyte.TaskEnvironment(name="named_dir_demo", depends_on=[container_env, list_env])


@env.task
async def main() -> str:
    # The local basename must survive the upload -> stage round-trip.
    local = Path("/tmp/genome.fasta")
    local.write_text(">seq1\nACGTACGTACGT\n")
    fasta = await File.from_local(str(local))
    # Expect the task to print: "genome.fasta".
    return await check_extension(fasta=fasta)


@env.task
async def main_list() -> str:
    d = Path("/tmp/reads")
    d.mkdir(exist_ok=True)
    (d / "sample_R1.fastq.gz").write_bytes(b"@r1\nACGT\n+\nIIII\n")
    (d / "sample_R2.fastq.gz").write_bytes(b"@r2\nTGCA\n+\nIIII\n")
    r1 = await File.from_local(str(d / "sample_R1.fastq.gz"))
    r2 = await File.from_local(str(d / "sample_R2.fastq.gz"))
    # Expect the task to print: "sample_R1.fastq.gz sample_R2.fastq.gz".
    return await check_reads(reads=[r1, r2])


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(main)
    print(run.url)
