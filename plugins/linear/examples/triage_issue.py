"""Triage Linear issues from Flyte tasks.

This example reads issues from a Linear team, auto-labels a newly created
issue by updating it, and comments back with the result.

Requirements:
    pip install flyteplugins-linear

Setup:
    flyte create secret LINEAR_API_KEY --value <api-key>

Usage:
    python plugins/linear/examples/triage_issue.py
"""

import flyte

from flyteplugins.linear import LinearClient

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-linear")

env = flyte.TaskEnvironment(
    name="linear-triage",
    image=image,
    secrets=[flyte.Secret("LINEAR_API_KEY", as_env_var="LINEAR_API_KEY")],
)


@env.task
async def summarize_backlog(team_key: str) -> str:
    """Read the team's backlog issues and summarize them."""
    async with LinearClient() as client:
        issues = await client.list_issues(team_key=team_key, state="Backlog")
    if not issues:
        return "backlog is empty"
    return "\n".join(f"{i['identifier']}: {i['title']}" for i in issues)


@env.task
async def triage_issue(issue_id: str) -> str:
    """Move an issue to In Progress and comment on it.

    This is the task the webhook example launches for every newly created
    issue. `issue_id` is the Linear UUID carried by webhook events.
    """
    async with LinearClient() as client:
        issue = await client.get_issue(issue_id)
        if issue["state"] != "In Progress":
            team_id = await _team_id_for(client, issue["team"])
            states = await client.list_workflow_states(team_id)
            in_progress = next((s for s in states if s["name"] == "In Progress"), None)
            if in_progress:
                await client.update_issue(issue_id, state_id=in_progress["id"])
        await client.add_comment(issue_id, "Flyte picked this issue up — investigating.")
    return f"triaged {issue['identifier']}"


async def _team_id_for(client: LinearClient, team_key: str) -> str:
    teams = await client.list_teams()
    for team in teams:
        if team["key"] == team_key:
            return team["id"]
    raise ValueError(f"team {team_key} not found")


if __name__ == "__main__":
    flyte.run(summarize_backlog, team_key="ENG")
