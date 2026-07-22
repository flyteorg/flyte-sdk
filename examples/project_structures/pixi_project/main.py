import pathlib

import flyte

env = flyte.TaskEnvironment(
    name="pixi_project_test",
    resources=flyte.Resources(memory="250Mi"),
    # The pixi environment (conda + PyPI packages, one solve) is installed at build
    # time and becomes the image's runtime environment. Run `pixi lock` next to the
    # manifest to get reproducible builds — the lock file is picked up automatically.
    image=flyte.Image.from_debian_base().with_pixi_project(
        manifest_file=pathlib.Path("pixi.toml"),
    ),
)


@env.task
def fn(x: int) -> int:  # Type annotations are recommended.
    import numpy as np

    slope, intercept = 2, 5
    return int(np.multiply(slope, x) + intercept)


@env.task
def main(x_list: list[int]) -> float:
    x_len = len(x_list)
    if x_len < 10:
        raise ValueError(f"x_list doesn't have a larger enough sample size, found: {x_len}")

    # flyte.map is like Python map, but runs in parallel.
    y_list = list(flyte.map(fn, x_list))
    y_mean = sum(y_list) / len(y_list)
    print(y_mean)
    return y_mean


if __name__ == "__main__":
    # Establish a remote connection from within your script.
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)

    # Run your tasks remotely inline and pass parameter data.
    run = flyte.run(main, x_list=list(range(10)))
    print(run.url)
