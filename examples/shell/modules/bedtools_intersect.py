from flyte.extras import shell
from flyte.io import File

IMAGE = "quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_0"

# Inputs use descriptive Python names where the bedtools CLI has case-only
# collisions (`-c`/`-C`, `-s`/`-S`, `-f`/`-F`). The shell renderer builds
# bash variable names by uppercasing the Python name, so two inputs whose
# names differ only in case would collide on the same `_FLAG_*` slot.
# `flag_aliases` maps each descriptive Python name to the actual CLI flag.
bedtools_intersect = shell.create(
    name="bedtools_intersect",
    image=IMAGE,
    inputs={
        "a": File,
        "b": list[File],
        "wa": bool | None,
        "wb": bool | None,
        "loj": bool | None,
        "wo": bool | None,
        "wao": bool | None,
        "u": bool | None,
        "count_overlaps": bool | None,
        "count_per_file": bool | None,
        "v": bool | None,
        "same_strand": bool | None,
        "opposite_strand": bool | None,
        "frac_a": float | None,
        "frac_b": float | None,
        "r": bool | None,
        "e": bool | None,
        "ubam": bool | None,
        "bed": bool | None,
        "sorted": bool | None,
        "nonamecheck": bool | None,
        "g": File | None,
        "names": str | None,
        "filenames": bool | None,
        "sortout": bool | None,
        "split": bool | None,
        "header": bool | None,
        "nobuf": bool | None,
        "iobuf": str | None,
    },
    outputs={"out": File},
    flag_aliases={
        "b": "-b",
        "count_overlaps": "-c",
        "count_per_file": "-C",
        "same_strand": "-s",
        "opposite_strand": "-S",
        "frac_a": "-f",
        "frac_b": "-F",
    },
    script=r"""
        bedtools intersect \
            -a {inputs.a} \
            {flags.b} \
            {flags.wa} {flags.wb} {flags.loj} {flags.wo} {flags.wao} \
            {flags.u} {flags.count_overlaps} {flags.count_per_file} {flags.v} \
            {flags.same_strand} {flags.opposite_strand} \
            {flags.frac_a} {flags.frac_b} {flags.r} {flags.e} \
            {flags.ubam} {flags.bed} \
            {flags.sorted} {flags.nonamecheck} {flags.g} \
            {flags.names} {flags.filenames} {flags.sortout} \
            {flags.split} {flags.header} {flags.nobuf} {flags.iobuf} \
            > {outputs.out}
    """,
)
