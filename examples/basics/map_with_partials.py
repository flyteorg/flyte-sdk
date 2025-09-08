from functools import partial

import flyte

env = flyte.TaskEnvironment(name="map_partial")


@env.task
def my_task(name: str, constant_param: str, batch_id: str) -> str:
    print(name, constant_param, batch_id)
    return name


@env.task
def my_task_correct(batch_id: str, name: str, constant_param: str) -> str:
    print(name, constant_param, batch_id)
    return name


@env.task
def single_param_task(a: int) -> str:
    return f"{a}"


@env.task
def main():
    compounds = [str(i) for i in range(3)]
    constant_param = "shared_config"

    print(list(flyte.map(single_param_task, range(3))), flush=True)

    task_with_constant_correct = partial(my_task_correct, constant_param=constant_param, name="daniel")
    # This should work, as batch_id is the first parameter and the only one left for mapping
    v = list(flyte.map(task_with_constant_correct, compounds, return_exceptions=False))
    print("\n".join(v), flush=True)

    try:
        # This should raise a TypeError, as batch_id is not the first parameter
        task_with_constant = partial(my_task, constant_param=constant_param, name="daniel")
        list(flyte.map(task_with_constant, compounds))
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")


@env.task
def raise_error_task(x: int) -> str:
    if x == 2:
        raise ValueError("Intentional error for testing")
    return f"Task {x}"


@env.task
def error_handling_main(n: int):
    results = list(flyte.map(raise_error_task, range(n), return_exceptions=False))
    for res in results:
        if isinstance(res, Exception):
            print(f"Error encountered: {res}")
        else:
            print(res)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(main)
    print(run.url)
