import asyncio
import time

import flyte
import flyte.errors

env = flyte.TaskEnvironment("external-abort")


@env.task
async def long_sleeper(sleep_for: float):
    await asyncio.sleep(sleep_for)


@env.task
async def main(n: int, sleep_for: float) -> str:
    coros = [long_sleeper(sleep_for) for _ in range(n)]
    results = await asyncio.gather(*coros, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, flyte.errors.ActionAbortedError):
            print(f"Action [{i}] was externally aborted")
    return "Hello World!"


if __name__ == "__main__":
    import flyte.models
    import flyte.remote

    flyte.init_from_config()
    r: flyte.remote.Run = flyte.run(main, 10, 30.0)
    print(r.url)

    def kill_one_action():
        while True:
            time.sleep(10.0)
            actions = list(
                flyte.remote.Action.listall(for_run_name=r.name, in_phase=[flyte.models.ActionPhase.RUNNING])
            )
            if len(actions) > 1:
                print(f"I can now see {len(actions)} actions")
                for a in actions:
                    if a.name != "a0":  # Ignore a0 to be aborted
                        print(f"I am killing action {a.name}")
                        a.abort("External abort!!")
                        return
            else:
                print("Waiting for child actions to be created")

    kill_one_action()
    print("Now I will wait for the run to complete")
    r.wait()
    v = r.outputs()
    assert v[0] == "Hello World!"
