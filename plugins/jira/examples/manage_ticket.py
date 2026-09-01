"""Open and progress Jira issues from Flyte tasks.

This example shows the basic client surface: creating a ticket, searching with
JQL, transitioning a ticket through its workflow, and commenting.

Requirements:
    pip install flyteplugins-jira

Setup:
    flyte create secret JIRA_BASE_URL --value https://<site>.atlassian.net
    flyte create secret JIRA_EMAIL --value you@example.com
    flyte create secret JIRA_API_TOKEN --value <api-token>

Usage:
    python plugins/jira/examples/manage_issue.py
"""

import flyte

from flyteplugins.jira import JiraClient

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-jira")

env = flyte.TaskEnvironment(
    name="jira-tickets",
    image=image,
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
    ],
)


@env.task
async def open_ticket(project_key: str, summary: str, description: str) -> str:
    """Create a ticket and return its URL."""
    async with JiraClient() as client:
        issue = await client.create_issue.aio(project_key, summary, description=description)
    return issue["url"]


@env.task
async def start_work(issue_key: str) -> str:
    """Transition a ticket to In Progress and comment on it."""
    async with JiraClient() as client:
        await client.transition_issue.aio(issue_key, "In Progress")
        await client.add_comment.aio(issue_key, "Flyte picked this ticket up.")
    return issue_key


@env.task
async def summarize_open_bugs(project_key: str) -> str:
    """Search open bugs in a project and summarize them."""
    async with JiraClient() as client:
        issues = await client.search_issues.aio(
            f"project = {project_key} AND issuetype = Bug AND statusCategory != Done ORDER BY priority DESC"
        )
    if not issues:
        return "no open bugs"
    return "\n".join(f"{i['key']} [{i['priority']}] {i['summary']}" for i in issues[:20])


@env.task
async def triage_issue(issue_key: str) -> str:
    """Comment on a newly created issue.

    This is the task `react_to_jira_events.py` launches for every
    `jira:issue_created` event.
    """
    async with JiraClient() as client:
        issue = await client.get_issue.aio(issue_key)
        await client.add_comment.aio(issue_key, f"Flyte triaged this issue (status: {issue.get('status')}).")
    return f"triaged {issue_key}"


if __name__ == "__main__":
    # Replace with a project key you can access.
    flyte.run(summarize_open_bugs, project_key="PROJ")
