"""Read and write Linear from tasks, with `gql` over Linear's GraphQL API.

Linear ships no official Python SDK; `gql` is the maintained GraphQL client and
Linear's API is a single endpoint, so there is little to wrap.

Requirements:
    pip install flyte "gql[httpx]"

Setup:
    flyte create secret LINEAR_API_KEY --value lin_api_...

Usage:
    flyte run examples/external_saas_integrations/linear_triage_issue.py \\
        summarize_backlog --team_key ENG
"""

import os

import flyte

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("gql[httpx]")

env = flyte.TaskEnvironment(
    name="linear-triage",
    image=image,
    secrets=[flyte.Secret("LINEAR_API_KEY", as_env_var="LINEAR_API_KEY")],
)

ENDPOINT = "https://api.linear.app/graphql"


async def _execute(query: str, variables: dict | None = None) -> dict:
    from gql import Client, gql
    from gql.transport.httpx import HTTPXAsyncTransport

    transport = HTTPXAsyncTransport(url=ENDPOINT, headers={"Authorization": os.environ["LINEAR_API_KEY"]})
    async with Client(transport=transport, fetch_schema_from_transport=False) as session:
        return await session.execute(gql(query), variable_values=variables or {})


@env.task
async def summarize_backlog(team_key: str, limit: int = 25) -> str:
    """List a team's open issues."""
    data = await _execute(
        """
        query Backlog($teamKey: String!, $first: Int!) {
          issues(first: $first, filter: {team: {key: {eq: $teamKey}},
                                         completedAt: {null: true},
                                         canceledAt: {null: true}}) {
            nodes { identifier title state { name } }
          }
        }
        """,
        {"teamKey": team_key, "first": limit},
    )
    nodes = data["issues"]["nodes"]
    return "\n".join(f"- {n['identifier']}: {n['title']} [{n['state']['name']}]" for n in nodes) or "backlog is empty"


@env.task
async def triage_issue(issue_id: str) -> str:
    """Comment on an issue by its UUID.

    This is what `webhook_receiver.py` launches for every `Issue.create`. Linear
    webhooks carry the entity UUID, which is what `commentCreate` wants.
    """
    await _execute(
        """
        mutation Comment($input: CommentCreateInput!) {
          commentCreate(input: $input) { success }
        }
        """,
        {"input": {"issueId": issue_id, "body": "Flyte triaged this issue."}},
    )
    return f"triaged {issue_id}"


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(summarize_backlog, team_key="ENG").url)
