"""bedtools — genome arithmetic CLI tools.

Wraps tools from the `bedtools suite <https://bedtools.readthedocs.io/>`_
shipped via the ``quay.io/biocontainers/bedtools`` image.

Currently exposed:

- :data:`bedtools_intersect` — report overlaps between two feature files.
- :data:`bedtools_sort` — sort BED/GFF/VCF features by chromosome and start.
- :data:`bedtools_merge` — combine overlapping or nearby features into one.

The module-level :data:`env` is a single :class:`flyte.TaskEnvironment`
containing every bedtools task. All commands share the same biocontainer
image, so they live in the same env. Pipelines depend on this one env to
gain access to every bedtools subcommand at once::

    from flyteplugins.bio.bedtools import (
        bedtools_intersect,
        bedtools_sort,
        bedtools_merge,
        env as bedtools_env,
    )

    env = flyte.TaskEnvironment(name="my_pipeline", depends_on=[bedtools_env])
"""

import flyte
from flyte.extras import shell
from flyte.io import File

# Pinned biocontainer URI. Update when bumping bedtools version.
BEDTOOLS_IMAGE = "quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_0"

bedtools_intersect = shell.create(
    name="bedtools_intersect",
    image=BEDTOOLS_IMAGE,
    inputs={
        "a": File,
        "b": list[File],
        # Output mode (mutually exclusive in practice)
        "wa": bool,
        "wb": bool,
        "loj": bool,
        "wo": bool,
        "wao": bool,
        "u": bool,
        "c": bool,
        "C": bool,
        "v": bool,
        # BAM output / BAM-to-BED conversion
        "ubam": bool,
        "bed_output": bool,  # mapped to -bed
        # Strand
        "s": bool,
        "S": bool,
        # Overlap fraction
        "f": float | None,
        "F": float | None,
        "r": bool,
        "e": bool,
        # Split BAM/BED12 entries
        "split": bool,
        # Sorted-input optimisation
        "sorted": bool,
        "g": File | None,
        "nonamecheck": bool,
        # Multi-database controls
        "names": str | None,  # space-separated aliases, one per -b file
        "filenames": bool,
        "sortout": bool,
        # Output control
        "header": bool,
        "nobuf": bool,
        "iobuf": str | None,  # e.g. "128M", "1G"
    },
    flag_aliases={
        "bed_output": "-bed",
    },
    defaults={
        "wa": False, "wb": False, "loj": False, "wo": False, "wao": False,
        "u": False, "c": False, "C": False, "v": False,
        "ubam": False, "bed_output": False,
        "s": False, "S": False,
        "r": False, "e": False,
        "split": False,
        "sorted": False, "nonamecheck": False,
        "filenames": False, "sortout": False,
        "header": False, "nobuf": False,
    },
    outputs={
        "bed": File,
    },
    script=r"""
        bedtools intersect \
            {flags.wa} {flags.wb} {flags.loj} {flags.wo} {flags.wao} \
            {flags.u} {flags.c} {flags.C} {flags.v} \
            {flags.ubam} {flags.bed_output} \
            {flags.s} {flags.S} \
            {flags.f} {flags.F} {flags.r} {flags.e} \
            {flags.split} \
            {flags.sorted} {flags.g} {flags.nonamecheck} \
            {flags.names} {flags.filenames} {flags.sortout} \
            {flags.header} {flags.nobuf} {flags.iobuf} \
            -a {inputs.a} \
            -b {inputs.b} \
            > {outputs.bed}
    """,
)

bedtools_sort = shell.create(
    name="bedtools_sort",
    image=BEDTOOLS_IMAGE,
    inputs={
        "i": File,
        # Sort orderings — `chrThenSizeA` and friends are bool flags.
        "header": bool,
        # Genome file: enforces a chromosome sort order across files.
        "g": File | None,
    },
    defaults={
        "header": False,
    },
    outputs={
        "sorted": File,
    },
    script=r"""
        bedtools sort \
            {flags.header} {flags.g} \
            -i {inputs.i} \
            > {outputs.sorted}
    """,
)


bedtools_merge = shell.create(
    name="bedtools_merge",
    image=BEDTOOLS_IMAGE,
    inputs={
        "i": File,  # Must be sorted.
        "s": bool,  # Require same strand.
        "S": str | None,  # Force one strand: "+" or "-".
        "d": int | None,  # Max distance between features to merge.
        "header": bool,
    },
    defaults={
        "s": False,
        "header": False,
    },
    outputs={
        "merged": File,
    },
    script=r"""
        bedtools merge \
            {flags.s} {flags.S} {flags.d} {flags.header} \
            -i {inputs.i} \
            > {outputs.merged}
    """,
)

env = flyte.TaskEnvironment.from_task(
    "bedtools",
    bedtools_intersect.as_task(),
    bedtools_sort.as_task(),
    bedtools_merge.as_task(),
)
