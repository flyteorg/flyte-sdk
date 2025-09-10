import flyte

env = flyte.TaskEnvironment("union_types")


@env.task
def complex_task(data: dict[str, list[str] | None] | None) -> dict[str, list[str] | None] | None:
    return data


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(complex_task, data={"key": ["value1", "value2"]})
    print(run)
    print(run.url)
