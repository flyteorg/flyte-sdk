# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte",
# ]
#
# [tool.pixi.workspace]
# channels = ["conda-forge"]
#
# [tool.pixi.dependencies]
# numpy = "*"
# ///
"""Build a task image from a pixi script.

`Image.from_pixi_script` is the pixi counterpart to `Image.from_uv_script`: the image is
described by the PEP 723 block at the top of this very file. What pixi adds over uv is the
conda side of the world — `[tool.pixi.dependencies]` pulls packages from conda channels,
which is how you get binaries (MKL builds, GDAL, CUDA toolkits) that are not on PyPI.

Note that `flyte` itself has to be listed in `dependencies`: unlike `from_debian_base`, a
script-defined image installs exactly what the script declares.

`[tool.pixi.workspace]` deliberately leaves `platforms` out here, so flyte fills it in from
the platforms the image is built for. Declare it explicitly if you want to pin the
resolution — for instance before running `pixi lock --script` to produce a
`pixi_image.py.pixi.lock` sidecar, which flyte then installs with `--locked`.

Run it with:

    python examples/image/pixi_image.py
"""

import flyte
from flyte import Image

image = Image.from_pixi_script(__file__, name="pixi-hello", registry="ghcr.io/flyteorg")

env = flyte.TaskEnvironment(name="pixi_hello", image=image)


@env.task
async def mean(values: list[float]) -> float:
    # numpy comes from the conda channel via [tool.pixi.dependencies].
    import numpy as np

    return float(np.mean(values))


@env.task
async def main(values: list[float] | None = None) -> str:
    values = values or [1.0, 2.0, 3.0, 4.0]
    return f"mean({values}) = {await mean(values)}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.name)
    print(run.url)
    run.wait()
