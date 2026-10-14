"""Read and write Jira from tasks, with the official `jira` package.

Requirements:
    pip install flyte jira

Setup:
    flyte create secret JIRA_BASE_URL  --value https://<site>.atlassian.net
    flyte create secret JIRA_EMAIL     --value you@example.com
    flyte create secret JIRA_API_TOKEN --value <api-token>

Usage:
    flyte run examples/external_saas_integrations/jira_manage_ticket.py \\
        open_ticket --project_key PROJ --summary "From Flyte"
"""

import os

import flyte

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("jira")

env = flyte.TaskEnvironment(
    name="jira-tickets",
    image=image,
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
    ],
)


def _client():
    from jira import JIRA

    return JIRA(
        server=os.environ["JIRA_BASE_URL"],
        basic_auth=(os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"]),
    )


@env.task
def open_ticket(project_key: str, summary: str, description: str = "") -> str:
    """Create an issue and return its key."""
    issue = _client().create_issue(
        project=project_key, summary=summary, description=description, issuetype={"name": "Task"}
    )
    return issue.key


@env.task
def triage_issue(issue_key: str) -> str:
    """Comment on a newly created issue.

    This is what `webhook_receiver.py` launches for every `jira:issue_created`.
    """
    jira = _client()
    issue = jira.issue(issue_key)
    jira.add_comment(issue, f"Flyte triaged this issue (status: {issue.fields.status.name}).")
    return f"triaged {issue_key}"


@env.task
def summarize_open_bugs(project_key: str, limit: int = 20) -> str:
    """Summarize open bugs via JQL."""
    issues = _client().search_issues(
        f'project = "{project_key}" AND issuetype = Bug AND statusCategory != Done', maxResults=limit
    )
    return "\n".join(f"- {i.key}: {i.fields.summary}" for i in issues) or "no open bugs"


@env.task
def start_work(issue_key: str, transition: str = "In Progress") -> str:
    """Transition an issue by name, listing the valid ones when it does not apply.

    Jira rejects transitions the issue's workflow does not offer, so resolve the
    name against what is actually available rather than guessing an id.
    """
    jira = _client()
    issue = jira.issue(issue_key)
    available = {t["name"].lower(): t["id"] for t in jira.transitions(issue)}
    target = available.get(transition.lower())
    if target is None:
        raise ValueError(f"{issue_key} cannot transition to {transition!r}; available: {sorted(available)}")
    jira.transition_issue(issue, target)
    return f"{issue_key} -> {transition}"


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(open_ticket, project_key="PROJ", summary="Flyte test issue").url)
