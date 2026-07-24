import flyte

image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse")

env1 = flyte.TaskEnvironment(
    name="test1",
    queue="jan-demo1",
    reusable=flyte.ReusePolicy(replicas=10),
    image=image,
)

env2 = flyte.TaskEnvironment(
    name="test2",
    queue="jan-demo2",
    reusable=flyte.ReusePolicy(replicas=10),
    image=image,
)


@env1.task
def fn1(x: int) -> int:
    slope, intercept = 2, 5
    print(slope * x + intercept)
    return slope * x + intercept


@env1.task
def main1(x_list: list[int] = list(range(100000))) -> float:
    x_len = len(x_list)
    if x_len < 10:
        raise ValueError(f"x_list doesn't have a larger enough sample size, found: {x_len}")

    y_list = list(flyte.map(fn1, x_list))
    y_mean = sum(y_list) / len(y_list)
    return y_mean


@env2.task
def fn2(x: int) -> int:
    slope, intercept = 2, 5
    print(slope * x + intercept)
    return slope * x + intercept


@env2.task
def main2(x_list: list[int] = list(range(100000))) -> float:
    x_len = len(x_list)
    if x_len < 10:
        raise ValueError(f"x_list doesn't have a larger enough sample size, found: {x_len}")

    y_list = list(flyte.map(fn2, x_list))
    y_mean = sum(y_list) / len(y_list)
    return y_mean


if __name__ == "__main__":
    flyte.init_from_config("dogfood.yaml")
    run = flyte.run(main2)
    print(run.url)
    run = flyte.run(main1)
    print(run.url)
