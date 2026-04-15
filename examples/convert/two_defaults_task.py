"""
Task with defaults: ``a=10``, ``b=0``.

Run:
  flyte run examples/convert/two_defaults_task.py repro --a=3 --b=1
  python examples/convert/two_defaults_task.py
"""

import flyte

env = flyte.TaskEnvironment(name="carina")


@env.task
async def porch_zero_default_repro(a: int = 10, b: int = 0) -> str:
    return f"a={a}, b={b}"


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(porch_zero_default_repro)
    print(r.name, r.url)
    r.wait()
