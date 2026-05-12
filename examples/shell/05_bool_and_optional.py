"""Bool flags, Optional inputs, and flag_aliases.

- ``bool`` inputs become CLI flags via ``{flags.<name>}`` — ``True`` emits
  the flag (default ``-name``); ``False`` collapses to empty.
- ``T | None`` inputs render to empty when ``None`` — so a missing
  optional silently drops out of the argv. Useful for tools where a flag
  may or may not apply.
- ``flag_aliases`` overrides per-input — needed when the Python name
  doesn't match the CLI flag (kebab-case, double dash, reserved words).

Run locally::

    uv run python 05_bool_and_optional.py
"""

import sys
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.extras import shell
from flyte.extras.shell import Stdout

# A faux "tool" — echo just shows the composed argv. In a real bio tool,
# the bool flags become e.g. -v, -w, --case-insensitive, etc.
report = shell.create(
    name="report",
    image="debian:12-slim",
    cache="disable",
    inputs={
        "title": str,
        # Booleans — emitted only when True.
        "verbose": bool,
        "case_insensitive": bool,
        # Optional scalar — emitted only when set.
        "threads": int | None,
    },
    outputs={"argv": Stdout(type=str)},
    flag_aliases={
        # Python name != CLI flag — override.
        "case_insensitive": "--case-insensitive",
        "threads": "--threads",
    },
    script=r"""
        echo report: \
            --title {inputs.title} \
            {flags.verbose} \
            {flags.case_insensitive} \
            {flags.threads}
    """,
)


env = flyte.TaskEnvironment(
    name="shell_bool_optional",
    depends_on=[report.env],
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


@env.task(cache="disable")
async def bool_demo() -> tuple[str, str, str]:
    # All flags on.
    a = await report(title="full", verbose=True, case_insensitive=True, threads=8)
    # Only verbose; threads omitted (None).
    b = await report(
        title="partial", verbose=True, case_insensitive=False, threads=None
    )
    # All flags off.
    c = await report(title="bare", verbose=False, case_insensitive=False, threads=None)
    return a, b, c


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"
    run = flyte.with_runcontext(mode=mode).run(bool_demo)
    print(run.url if mode == "remote" else run)
    print(f"Outputs:\n{run.outputs()}")
