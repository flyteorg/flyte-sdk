from typing import Union

import flyte

env = flyte.TaskEnvironment(
    name="union_type_demo",
)


@env.task
def process_union_input(input_val: Union[str, None]) -> str:
    if input_val is None:
        return "Input was None!"
    return f"Input was a string: {input_val}"


@env.task
def union_type_simple_task(input_val: Union[str, None] = None) -> str:
    return process_union_input(input_val)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(union_type_simple_task, input_val="hello world")
    print(run.name)
    print(run.url)
    run.wait()
