from pathlib import Path

from my_type import PositiveInt

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    image=flyte.Image.from_debian_base().with_uv_project(
        Path(__file__).parent / "my_transformer/pyproject.toml", pre=True
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


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)

    positive_ints = [PositiveInt(i) for i in range(1, 5)]
    run = flyte.run(main, x_list=positive_ints)

    print(run.name)
    print(run.url)
