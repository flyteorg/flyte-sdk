import flyte
import flyte.dsl as dsl

env = flyte.TaskEnvironment("compiled")


@env.task
def add(a: int, b: int) -> int:
    return a + b


@env.task
def multiply(a: int, b: int) -> int:
    return a * b


@env.task
def print_val(value: int):
    print(f"Result: {value}")


def workflow(x: int, y: int) -> int:
    print_val(x)
    print_val(y)
    sum_result = add(x, y)
    product_result = multiply(x, y)
    final_result = add(sum_result, product_result)
    print_val(final_result)
    return final_result


@env.task
def main(x: int, y: int) -> int:
    dag = dsl.compile(workflow)
    print(dag)
    v = dsl.run(dag, x=x, y=y)
    if v is None:
        raise ValueError("Workflow did not return a value")
    return v


if __name__ == "__main__":
    flyte.init_from_config()
    result = flyte.run(main, x=3, y=4)
    print(f"Workflow URL: {result.url}")
    print(f"Final Result: {result.outputs()}")
