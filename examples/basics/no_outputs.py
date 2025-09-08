import flyte

env = flyte.TaskEnvironment(name="no_outputs")


@env.task
async def no_outputs_task():
    print("This task does not return any outputs.")


@env.task
async def main():
    await no_outputs_task()
    await no_outputs_task()
    await no_outputs_task()


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    # flyte.init()
    run = flyte.run(main)
    print(run.url)
