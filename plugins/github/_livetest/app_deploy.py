"""Live deployment of GitHubAppEnvironment (react_to_pr_events.py equivalent)."""

import flyte

from _livetest.common import GH_SECRET, WEBHOOK_SECRET, image
from flyteplugins.github import GitHubAppEnvironment, launch_task

app_env = GitHubAppEnvironment(
    name="github-integration",
    image=image("fastapi>=0.115", "uvicorn>=0.30"),
    secrets=[GH_SECRET, WEBHOOK_SECRET],
    repos=[],
    # GitHub cannot authenticate to the Flyte platform; the plugin's own
    # HMAC signature check is the auth mechanism for /webhook.
    requires_auth=False,
)


@app_env.on_event("pull_request.opened")
async def triage_new_pr(event):
    import flyte.remote as remote

    from flyteplugins.github import DuplicateRun

    task = remote.Task.get(name="triage_pr", auto_version="latest")
    try:
        run = launch_task(task, key=event.dedupe_key(), repo=event.repository, number=event.number)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


@app_env.on_event("issues.opened")
async def label_new_issues(event):
    from flyteplugins.github import GitHubClient

    if event.repository is None or event.number is None:
        return None
    async with GitHubClient() as client:
        await client.add_labels(event.repository, event.number, ["flyte-triage"])
    return {"labeled": event.number}


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(app_env)
    handle.activate(wait=True)
    print("ENDPOINT:", handle.endpoint)
