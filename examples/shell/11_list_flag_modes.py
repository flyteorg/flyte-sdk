"""list[File] flag rendering modes: join, repeat, and comma.

`list[File]` inputs can be rendered under a CLI flag in three ways:

- `join` (default): `-I a.txt b.txt c.txt`
- `repeat`: `-I a.txt -I b.txt -I c.txt`
- `comma`: `--inputs a.txt,b.txt,c.txt`

This example prints the argv tokens exactly as the tool sees them, so the
differences between the modes are easy to inspect.

Run locally::

    uv run python 11_list_flag_modes.py
"""

import sys
import tempfile
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.extras import shell
from flyte.extras.shell import Stdout
from flyte.io import File


def _argv_dump_task(name: str, flag_alias: str | tuple[str, str]):
    return shell.create(
        name=name,
        image="debian:12-slim",
        cache="disable",
        inputs={"parts": list[File]},
        outputs={"argv": Stdout(type=str)},
        flag_aliases={"parts": flag_alias},
        script=r"""
            set -- {flags.parts}
            printf '%s\n' "$@"
        """,
    )


show_join = _argv_dump_task("show_join_mode", "-I")
show_repeat = _argv_dump_task("show_repeat_mode", ("-I", "repeat"))
show_comma = _argv_dump_task("show_comma_mode", ("--inputs", "comma"))


env = flyte.TaskEnvironment(
    name="shell_list_flag_modes",
    depends_on=[show_join.env, show_repeat.env, show_comma.env],
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
async def list_mode_demo(parts: list[File]) -> tuple[str, str, str]:
    join_argv = await show_join(parts=parts)
    repeat_argv = await show_repeat(parts=parts)
    comma_argv = await show_comma(parts=parts)
    return join_argv, repeat_argv, comma_argv


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    parts: list[File] = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{i}.txt", delete=False
        ) as f:
            f.write(f"part-{i}\n")
            parts.append(File.from_local_sync(f.name))

    run = flyte.with_runcontext(mode=mode).run(list_mode_demo, parts)
    print(run.url if mode == "remote" else run)
    print(f"Outputs:\n{run.outputs()}")
