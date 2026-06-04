"""The ``bedtools intersect`` wrapper."""

from __future__ import annotations

from flyte.extras import shell
from flyte.io import File

from ._shared import BEDTOOLS_IMAGE

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
        "wa": False,
        "wb": False,
        "loj": False,
        "wo": False,
        "wao": False,
        "u": False,
        "c": False,
        "C": False,
        "v": False,
        "ubam": False,
        "bed_output": False,
        "s": False,
        "S": False,
        "r": False,
        "e": False,
        "split": False,
        "sorted": False,
        "nonamecheck": False,
        "filenames": False,
        "sortout": False,
        "header": False,
        "nobuf": False,
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
