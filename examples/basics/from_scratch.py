from typing import Optional

import flyte

env = flyte.TaskEnvironment(name="test_env")


@env.task
def main(name: Optional[str] = None):
    print(f"Hello {name}")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, "xyz")
    print(run.url)
