from typing import Optional

import flyte

env = flyte.TaskEnvironment(name="test_env")


@env.task
def main(name: Optional[str] = None):
    print(f"Hello {name}")


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(main, "xyz")
    print(run.url)
