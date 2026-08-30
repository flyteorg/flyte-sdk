"""GitHub integration for Flyte.

Read and write GitHub from Flyte tasks, react to GitHub webhook events through
an app environment, gate workflows on human PR reviews via `flyte.new_condition`,
and expose the operations as an MCP server for agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-github[app,mcp]"
```

## Read/write from tasks

```python
import flyte
from flyteplugins.github import GitHubClient

env = flyte.TaskEnvironment(
    name="github-demo",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-github"),
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
)

@env.task
async def summarize_pr(repo: str, number: int) -> str:
    async with GitHubClient() as client:
        pr = await client.get_pull_request(repo, number)
        files = await client.get_pull_request_files(repo, number)
    return f"{pr['title']}: {len(files)} files changed"
```

## Human review gate (condition with a JSON payload)

```python
from flyteplugins.github import review_pr

@env.task
async def gated_merge(repo: str, number: int) -> str:
    decision = await review_pr(repo, number)
    if decision.is_approved:
        async with GitHubClient() as client:
            await client.merge_pull_request(repo, number, merge_method="squash")
        return "merged"
    return f"blocked: {decision.summary}"
```

## React to GitHub events

```python
import flyte
from flyteplugins.github import GitHubAppEnvironment, launch_task

app_env = GitHubAppEnvironment(
    name="github-integration",
    secrets=[
        flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN"),
        flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET"),
    ],
)

@app_env.on_event("pull_request.opened")
async def triage_new_pr(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_pr", auto_version="latest")
    run = launch_task(task, key=event.dedupe_key(), repo=event.repository, number=event.number)
    return {"run": run.name}

flyte.serve(app_env)
```

The app's dashboard (`/`) walks through token creation, Flyte secret creation,
and repository webhook configuration.

## MCP server for agents

```python
import flyte
from flyteplugins.github import github_mcp_app_env

mcp_env = github_mcp_app_env("github-mcp")  # read-only by default
flyte.serve(mcp_env)
```
"""

from ._app import GitHubAppEnvironment
from ._client import GitHubClient
from ._config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_TOKEN_ENV_VAR,
    DEFAULT_WEBHOOK_SECRET_ENV_VAR,
    Config,
    default_config,
)
from ._dispatch import DUPE_LABEL_KEY, DuplicateRun, blocking_run, launch_task, run_name_for
from ._errors import GitHubAPIError, GitHubPluginError, MissingCredentialsError, WebhookSignatureError
from ._mcp import build_mcp_server, github_mcp_app_env
from ._review import (
    ReviewComment,
    ReviewContext,
    ReviewDecision,
    build_review_prompt,
    collect_review_context,
    parse_review_payload,
    review_pr,
)
from ._tools import TOOL_GROUPS, TOOL_REGISTRY, ToolInfo, build_tool_functions
from ._webhook import GitHubEvent, parse_webhook, verify_webhook_signature

__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_TOKEN_ENV_VAR",
    "DEFAULT_WEBHOOK_SECRET_ENV_VAR",
    "DUPE_LABEL_KEY",
    "TOOL_GROUPS",
    "TOOL_REGISTRY",
    "Config",
    "DuplicateRun",
    "GitHubAPIError",
    "GitHubAppEnvironment",
    "GitHubClient",
    "GitHubEvent",
    "GitHubPluginError",
    "MissingCredentialsError",
    "ReviewComment",
    "ReviewContext",
    "ReviewDecision",
    "ToolInfo",
    "WebhookSignatureError",
    "blocking_run",
    "build_mcp_server",
    "build_review_prompt",
    "build_tool_functions",
    "collect_review_context",
    "default_config",
    "github_mcp_app_env",
    "launch_task",
    "parse_review_payload",
    "parse_webhook",
    "review_pr",
    "run_name_for",
    "verify_webhook_signature",
]
