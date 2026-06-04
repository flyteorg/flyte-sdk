"""The ``bedtools merge`` wrapper."""

from __future__ import annotations

from flyte.extras import shell
from flyte.io import File

from ._shared import BEDTOOLS_IMAGE

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
