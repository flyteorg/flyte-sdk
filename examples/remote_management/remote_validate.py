import flyte
import flyte.errors
from flyte.remote import Task

env = flyte.TaskEnvironment(
    name="remote-validate-env",
    resources=flyte.Resources(memory="500Mi", cpu=1),
    image=flyte.Image.from_debian_base(),
)


@env.task
async def check_reference_task_exists(task_name: str, project: str | None = None, domain: str | None = None) -> bool:
    """
    Check if a remote task exists without executing it.

    Args:
        task_name: Name of the task (e.g., "my_env.my_task")
        project: Project name (optional, uses config default)
        domain: Domain name (optional, uses config default)

    Returns:
        True if task exists, False otherwise
    """
    try:
        # Get the lazy entity - this doesn't fetch yet
        lazy_task = Task.get(task_name, project=project, domain=domain, auto_version="latest")

        # Explicitly fetch to trigger the validation - this is where RemoteTaskNotFoundError is raised
        task_details = await lazy_task.fetch.aio()

        print(f"✓ Task '{task_name}' exists")
        print(f"  - Version: {task_details.version}")
        print(f"  - Type: {task_details.task_type}")
        print(f"  - Required args: {task_details.required_args}")
        return True

    except flyte.errors.RemoteTaskNotFoundError as e:
        # Task doesn't exist in the backend
        print(f"✗ Task '{task_name}' not found: {e}")
        print(f"  - Error code: {e.code}")
        print(f"  - Error kind: {e.kind}")
        return False
    except Exception as e:
        # Other errors (network, auth, etc.)
        print(f"✗ Error checking task '{task_name}': {type(e).__name__}: {e}")
        return False


# Example usage
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        check_reference_task_exists,
        task_name="penguin_training.training_pipeline",
        project="flytesnacks",
        domain="development",
    )
    print(f"Execution URL: {run.url}")
