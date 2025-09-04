from functools import partial

import flyte

env = flyte.TaskEnvironment(name="single_cpu", resources=flyte.Resources(cpu="1"))


@env.task
def my_task(name: str, constant_param: str, batch_id: int) -> str:
    print(name, constant_param, batch_id)
    return name


@env.task
def my_task_correct(batch_id: int, name: str, constant_param: str) -> str:
    print(name, constant_param, batch_id)
    return name


@env.task
def main():
    compounds = [str(i) for i in range(3)]
    constant_param = "shared_config"
    task_with_constant = partial(my_task, constant_param=constant_param, name="daniel")

    task_with_constant_correct = partial(my_task_correct, constant_param=constant_param, name="daniel")
    # This should work
    list(flyte.map(task_with_constant_correct, compounds))

    try:
        list(flyte.map(task_with_constant, compounds))
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(main)
    print(run.url)
