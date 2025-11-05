from pathlib import Path

from my_type import PositiveInt

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    image=flyte.Image.from_debian_base().with_uv_project(
        Path(__file__).parent / "my_transformer/pyproject.toml",
        pre=True,
        project_install_mode="install_project",
    ),
)


@env.task
def fn(x: PositiveInt) -> PositiveInt:
    slope, intercept = 2, 5
    result = slope * x.value + intercept
    return PositiveInt(result)


@env.task
def main(x_list: list[PositiveInt]) -> float:
    y_list = list(flyte.map(fn, x_list))
    y_mean = sum(y.value for y in y_list) / len(y_list)
    return y_mean

# Note the CLI doesn't work because we don't expose CLI transformers yet, this is an area for extension in the future.
# Does not work: flyte -vvv -c ~/.flyte/demo.yaml run main.py main --x_list '[1,3,5]'
if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)

    positive_ints = [PositiveInt(i) for i in range(1, 5)]
    run = flyte.run(main, x_list=positive_ints)

    print(run.name)
    print(run.url)
