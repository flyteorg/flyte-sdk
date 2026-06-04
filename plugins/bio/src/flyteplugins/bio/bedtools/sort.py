"""The ``bedtools sort`` wrapper."""

from __future__ import annotations

from flyte.extras import shell
from flyte.io import File

from ._shared import BEDTOOLS_IMAGE

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
