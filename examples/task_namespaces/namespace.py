"""Select an existing, operator-allowed Kubernetes namespace for a task.

Requires the companion backend change and executor.allowedTaskNamespaces to
include gpu-user-123. Provision the namespace and workload credentials first.
"""

import flyte
from flyte.extras import ContainerTask

env = flyte.TaskEnvironment(name="namespace_example")


@env.task(namespace="gpu-user-123")
async def train() -> str:
    return "Task workload placed in gpu-user-123"


# A container/script can select its namespace in the same way.
run_script = ContainerTask(
    name="training_script",
    image="my-training-image:latest",
    command=["python", "/app/train.py"],
    namespace="gpu-user-123",
)

container_env = flyte.TaskEnvironment.from_task("training_script_env", run_script)
