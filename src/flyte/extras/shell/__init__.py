"""Shell task — wrap a CLI tool packaged in a container image.

Designed as the foundation for bio module libraries (bedtools, samtools,
bcftools, GATK, etc.) and any other case where a user wants to call a
pre-built binary in a published container with typed inputs and outputs.

Compared to :class:`flyte.extras.ContainerTask`, this layer adds:

- A Python ``str.format``-style template surface (``{inputs.x}``, ``{flags.x}``,
  ``{outputs.x}``) instead of ``{{.inputs.x}}`` syntax.
- A closed type vocabulary for inputs: ``File``, ``Dir``, ``list[File]``,
  ``dict[str, str]``, scalars (``int`` / ``float`` / ``str``), ``bool``,
  and ``T | None`` of any of those.
- ``flags.<name>`` rendering: bool inputs become ``-name`` / ``""``,
  scalar inputs become ``-name value``, list inputs render with one of
  three modes (``join`` / ``repeat`` / ``comma``), dict inputs with one
  of two (``pairs`` / ``equals``).
- Output declarations use **bare types** for the common cases —
  ``File``, ``Dir``, ``int`` / ``float`` / ``str`` / ``bool`` — and
  three small collector classes for the cases that need extra semantics:

  * :class:`Glob` — pattern-filtered ``list[File]`` (the script writes
    files into ``/var/outputs/<name>/`` and the wrapper unpacks).
  * :class:`Stdout` / :class:`Stderr` — wrapper redirects the
    corresponding stream straight to ``/var/outputs/<name>``.

``Glob`` has two observable shapes:

- on the serialized task / remote wire interface, it is a ``Dir``
- when you call the Python shell wrapper (``await my_shell_task(...)``),
  that ``Dir`` is unpacked back into ``list[File]``

Example::

    from flyte.extras.shell import create, Glob
    from flyte.io import File

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
"""

from __future__ import annotations

from ._runtime import create
from ._types import (
    DictMode,
    FlagSpec,
    Glob,
    OutputSpec,
    Stderr,
    Stdout,
    listMode,
)

__all__ = [
    "DictMode",
    "FlagSpec",
    "Glob",
    "OutputSpec",
    "Stderr",
    "Stdout",
    "create",
    "listMode",
]
