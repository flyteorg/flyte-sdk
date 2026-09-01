"""Open and progress ClickUp tickets from Flyte tasks.

This example shows the basic client surface: creating a ticket, validating a
status transition against the list's workflow, moving the ticket, and
commenting on it.

Requirements:
    pip install flyteplugins-clickup

Setup:
    flyte create secret CLICKUP_TOKEN --value <token>

Usage:
    python plugins/clickup/examples/manage_ticket.py
"""

import flyte

from flyteplugins.clickup import ClickUpClient

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-clickup")

env = flyte.TaskEnvironment(
    name="clickup-tickets",
    image=image,
    secrets=[flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN")],
)


@env.task
async def open_ticket(list_id: str, name: str, description: str) -> str:
    """Create a ticket and return its URL."""
    async with ClickUpClient() as client:
        task = await client.create_task.aio(list_id, name, description=description)
    return task["url"]


@env.task
async def close_ticket(task_id: str, done_status: str = "done") -> str:
    """Move a ticket to a Done-like status, validating it first.

    ClickUp rejects transitions to statuses the ticket's list does not define,
    so the task checks `list_statuses` before updating.
    """
    async with ClickUpClient() as client:
        task = await client.get_task.aio(task_id)
        valid = await client.list_statuses.aio(task["list_id"])
        if done_status not in valid:
            raise ValueError(f"status {done_status!r} is not valid for this list; choose from {valid}")
        await client.update_task.aio(task_id, status=done_status)
        await client.add_comment.aio(task_id, "Closed by Flyte.")
    return task_id


@env.task
async def triage_task(task_id: str) -> str:
    """Comment on a newly created task.

    This is the task `react_to_clickup_events.py` launches for every
    `taskCreated` event.
    """
    async with ClickUpClient() as client:
        task = await client.get_task.aio(task_id)
        await client.add_comment.aio(task_id, f"Flyte triaged this ticket (status: {task.get('status')}).")
    return f"triaged {task_id}"


if __name__ == "__main__":
    # Replace with a list id from your ClickUp workspace.
    flyte.run(open_ticket, list_id="LIST_ID", name="Test ticket", description="Created by Flyte.")
