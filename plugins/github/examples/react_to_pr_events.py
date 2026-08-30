"""React to GitHub webhook events with the plugin's app environment.

`GitHubAppEnvironment` serves a setup dashboard (`/`) and a verified webhook
receiver (`/webhook`). This example registers a handler for newly opened pull
requests that launches an idempotent triage run.

Requirements:
    pip install "flyteplugins-github[app]"

Setup:
    flyte create secret GITHUB_TOKEN --value <token>
    flyte create secret GITHUB_WEBHOOK_SECRET --value <random-string>

    Then deploy this app:
        python plugins/github/examples/react_to_pr_events.py
    Open the app's dashboard, follow the setup instructions, and point a
    repository webhook at `<app-url>/webhook`.

Usage:
    python plugins/github/examples/react_to_pr_events.py
"""

import flyte

from flyteplugins.github import GitHubAppEnvironment, launch_task

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-github[app]")

app_env = GitHubAppEnvironment(
    name="github-integration",
    image=image,
    secrets=[
        flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN"),
        flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET"),
    ],
    # Only react to events from these repositories (empty = all).
    repos=[],
)


@app_env.on_event("pull_request.opened")
async def triage_new_pr(event):
    """Launch the triage task once per newly opened PR.

    The task must already be deployed (register the triage_pr task from
    read_write_pr.py or your own workflow first). `launch_task` dedupes on the
    event, so webhook redeliveries never launch a second run.
    """
    import flyte.remote as remote

    from flyteplugins.github import DuplicateRun

    task = remote.Task.get(name="triage_pr", auto_version="latest")
    try:
        run = launch_task(
            task,
            key=event.dedupe_key(),
            repo=event.repository,
            number=event.number,
        )
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


@app_env.on_event("issues.opened")
async def label_new_issues(event):
    """Auto-label new issues from the webhook, without launching a run."""
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
    print(f"Dashboard ready at {handle.endpoint}")
