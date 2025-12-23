from lib.workflows import env
from lib.workflows import utils

from flyte import task, workflow


@task
def another_task(message: str) -> str:
    """Another example task."""
    processed = utils.process_data(message)
    debug_mode = env.ENV_CONFIG["debug"]

    if debug_mode:
        return f"[DEBUG] {processed}"
    return processed


@workflow
def another_workflow(msg: str = "Hello") -> str:
    """Another example workflow."""
    return another_task(message=msg)


if __name__ == "__main__":
    print(another_workflow(msg="Test"))
