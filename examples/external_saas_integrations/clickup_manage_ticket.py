"""Read and write ClickUp from tasks, with `httpx` against its REST v2 API.

ClickUp ships no official Python SDK and the community ones are thin and
sporadically maintained, so this calls the REST API directly with `httpx`
rather than adding a dependency that wraps four endpoints.

Requirements:
    pip install flyte httpx

Setup:
    flyte create secret CLICKUP_TOKEN --value pk_...

Usage:
    flyte run examples/external_saas_integrations/clickup_manage_ticket.py \\
        open_ticket --list_id <list-id> --name "From Flyte"
"""

import os

import flyte

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("httpx")

env = flyte.TaskEnvironment(
    name="clickup-tickets",
    image=image,
    secrets=[flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN")],
)

API = "https://api.clickup.com/api/v2"


def _client():
    import httpx

    return httpx.AsyncClient(base_url=API, headers={"Authorization": os.environ["CLICKUP_TOKEN"]}, timeout=30)


@env.task
async def open_ticket(list_id: str, name: str, description: str = "") -> str:
    """Create a task and return its URL."""
    async with _client() as client:
        response = await client.post(f"/list/{list_id}/task", json={"name": name, "description": description})
        response.raise_for_status()
        return response.json()["url"]


@env.task
async def triage_task(task_id: str) -> str:
    """Comment on a newly created task.

    This is what `webhook_receiver.py` launches for every `taskCreated`.
    """
    async with _client() as client:
        task = await client.get(f"/task/{task_id}")
        task.raise_for_status()
        status = (task.json().get("status") or {}).get("status")
        posted = await client.post(
            f"/task/{task_id}/comment", json={"comment_text": f"Flyte triaged this ticket (status: {status})."}
        )
        posted.raise_for_status()
    return f"triaged {task_id}"


@env.task
async def close_ticket(task_id: str, done_status: str = "done") -> str:
    """Move a ticket to a Done-like status, validating it first.

    ClickUp rejects transitions to statuses the ticket's list does not define,
    with an opaque 400 — so check the list's statuses before trying.
    """
    async with _client() as client:
        task = await client.get(f"/task/{task_id}")
        task.raise_for_status()
        list_id = (task.json().get("list") or {}).get("id")

        listing = await client.get(f"/list/{list_id}")
        listing.raise_for_status()
        valid = [s["status"] for s in listing.json().get("statuses", [])]
        if done_status not in valid:
            raise ValueError(f"status {done_status!r} is not defined on list {list_id}; valid: {valid}")

        updated = await client.put(f"/task/{task_id}", json={"status": done_status})
        updated.raise_for_status()
    return f"{task_id} -> {done_status}"


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(open_ticket, list_id="LIST_ID", name="Flyte test ticket").url)
