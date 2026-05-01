import flyte

env = flyte.TaskEnvironment(name="user_logging_example")


@env.task
def process(name: str, iterations: int) -> str:
    flyte.logger.info("Starting process for name=%s, iterations=%d", name, iterations)

    results = []
    for i in range(iterations):
        flyte.logger.debug("Iteration %d of %d", i + 1, iterations)
        results.append(f"{name}_{i}")

    flyte.logger.info("Finished process, produced %d results", len(results))
    return ", ".join(results)


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.run(process, name="hello", iterations=3)