"""dict[str, str] inputs — pairs and equals rendering modes.

Two CLI conventions for "many key/value flags":

- ``pairs`` (default) — ``--memory 4G --threads 8`` (each k, each v an argv token).
  Use when the user puts the flag prefix in the dict key itself.
- ``equals`` — ``INPUT=a.bam INPUT=b.bam`` (Picard style). Each k=v becomes one
  argv token; the optional flag in ``flag_aliases`` is a per-key prefix.

Values can contain spaces, tabs, single quotes — all safe (transparently
encoded with ``\\x1e`` on the wire, decoded into a bash array at runtime).

Recipes — things that look like they need a richer dict but don't:

- **Bool as a CLI switch** (``--verbose``) → declare a separate ``bool`` input
  and use ``{flags.verbose}``. See ``05_bool_and_optional.py``.
- **Bool as a value string** (``REMOVE_DUPLICATES=true``) → already works;
  the value in ``dict[str, str]`` is just the string ``"true"`` (see the
  ``picard_style`` task below).
- **List of values under a repeated flag** (``-I a.bam -I b.bam``) → declare
  ``list[File]`` with ``flag_aliases={"name": ("-I", "repeat")}``.
- **List of strings, comma-joined** (``--exclude a,b,c``) → pass a
  pre-joined string yourself: ``extras={"--exclude": "a,b,c"}`` (see the
  ``comma_joined_extra`` task below).

Resist the urge to extend ``dict[str, str]`` to mixed value types — declaring
inputs individually gives you better type hints, IDE autocomplete, and clearer
error messages.

Run locally::

    uv run python 04_dict_inputs.py
"""

import sys
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.extras import shell
from flyte.extras.shell import Stdout

# `pairs` mode (default) — dict expanded as `key1 v1 key2 v2 …`.
echo_pairs = shell.create(
    name="echo_pairs",
    image="debian:12-slim",
    inputs={"opts": dict[str, str]},
    outputs={"argv": Stdout(type=str)},
    flag_aliases={
        "opts": ""
    },  # pairs mode is the default; no per-key prefix; keys already include `--`
    script=r"""
        echo {flags.opts}
    """,
)


# `equals` mode — Picard-style `KEY=value` tokens; `flag` is the prefix.
echo_equals = shell.create(
    name="echo_equals",
    image="debian:12-slim",
    inputs={"params": dict[str, str]},
    outputs={"argv": Stdout(type=str)},
    flag_aliases={"params": ("", "equals")},  # no prefix; each k=v as one token
    script=r"""
        echo {flags.params}
    """,
)


# Recipe: Picard-style with bool-as-value. The dict value is the string
# "true" — no special encoding, just a stringified bool.
picard_style = shell.create(
    name="picard_style",
    image="debian:12-slim",
    inputs={"params": dict[str, str]},
    outputs={"argv": Stdout(type=str)},
    flag_aliases={"params": ("", "equals")},
    script=r"""
        echo {flags.params}
    """,
)


# Recipe: comma-joined list value. The dict value is the user's pre-joined
# string; the script splits it (or hands it to a tool that accepts CSVs).
comma_joined_extra = shell.create(
    name="comma_joined_extra",
    image="debian:12-slim",
    inputs={"extras": dict[str, str]},
    outputs={"argv": Stdout(type=str)},
    flag_aliases={"extras": ""},
    script=r"""
        echo {flags.extras}
    """,
)


env = flyte.TaskEnvironment(
    name="shell_dict_inputs",
    depends_on=[echo_pairs.env, echo_equals.env, picard_style.env, comma_joined_extra.env],
    image=(
        flyte.Image.from_debian_base().clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent.parent / "dist",
                package_name="flyte",
            ),
            name="shell-basic",
        )
    ),
)


@env.task
async def dict_demo() -> tuple[str, str, str, str]:
    pairs_argv = await echo_pairs(
        opts={
            "--memory": "4G",
            "--threads": "8",
            "--label": "it's a test",      # single quote — safe via positional args
        }
    )
    equals_argv = await echo_equals(
        params={
            "INPUT": "a.bam",
            "OUTPUT": "b.bam",
            "METRICS": "metrics.txt",
        }
    )

    # Recipe — Picard-style with bool-as-value. The "true" / "false" are
    # plain strings; the dict[str, str] vocabulary already covers this.
    picard_argv = await picard_style(
        params={
            "INPUT": "in.bam",
            "OUTPUT": "out.bam",
            "REMOVE_DUPLICATES": "true",  # bool as value-string, not a switch
            "ASSUME_SORTED": "true",
        }
    )

    # Recipe — comma-joined "list of strings". User joins on the Python side;
    # the dict value is a single CSV string the script (or downstream tool)
    # can split.
    csv_argv = await comma_joined_extra(
        extras={
            "--include": "chr1,chr2,chr3",
            "--exclude": "chrM,chrY",
        }
    )

    return pairs_argv, equals_argv, picard_argv, csv_argv


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"
    run = flyte.with_runcontext(mode=mode).run(dict_demo)
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")
