import flyte

env = flyte.TaskEnvironment(
    name="from_scratch",
)


@env.task
def square(x: int) -> int:
    return x * x


@env.task
def main(n: int = 10) -> int:
    results = list(flyte.map(square, range(n)))
    return sum(results)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main, 10)
    print(r.url)
