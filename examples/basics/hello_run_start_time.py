import flyte

env = flyte.TaskEnvironment(
    name="hello_run_start_time",
)


@env.task()
async def hello_run_start_time() -> str:
    ctx = flyte.ctx()
    assert ctx is not None
    # run_start_time is populated by the backend from the run's start time (for scheduled-trigger
    # runs, the trigger's fire time) via the {{.runStartTime}} container arg. For ad-hoc runs it
    # defaults to the run-creation time.
    run_start_time = ctx.run_start_time
    print(f"Run start time: {run_start_time.isoformat()}")
    return run_start_time.isoformat()


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(hello_run_start_time)
    print(run.name)
    print(run.url)
    run.wait()
