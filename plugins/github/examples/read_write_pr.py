"""Read and write GitHub from Flyte tasks.

This example shows the basic client surface: reading pull requests and their
files, commenting, labeling, and reporting a check run. The `triage_pr` task
is the one the webhook example (`react_to_pr_events.py`) launches when a PR
opens.

Requirements:
    pip install flyteplugins-github

Setup:
    flyte create secret GITHUB_TOKEN --value <token-with-repo-scope>

Usage:
    python plugins/github/examples/read_write_pr.py
"""

import flyte

from flyteplugins.github import GitHubClient

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-github")

env = flyte.TaskEnvironment(
    name="github-read-write",
    image=image,
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
)


@env.task
async def summarize_pr(repo: str, number: int) -> str:
    """Read a pull request and summarize what it changes."""
    async with GitHubClient() as client:
        pr = await client.get_pull_request(repo, number)
        files = await client.get_pull_request_files(repo, number)
    summary = "\n".join(f"- {f['filename']} (+{f['additions']}/-{f['deletions']})" for f in files[:20])
    return f"{pr['title']} ({pr['head']} -> {pr['base']})\n{summary}"


@env.task
async def triage_pr(repo: str, number: int) -> str:
    """Label a new PR, comment on it, and report a check run.

    This is the task the webhook example launches for every newly opened PR.
    """
    async with GitHubClient() as client:
        pr = await client.get_pull_request(repo, number)
        await client.add_labels(repo, number, ["flyte-triage"])
        await client.create_issue_comment(
            repo,
            number,
            f"Flyte triage: this PR touches {pr.get('changed_files', '?')} files "
            f"(+{pr.get('additions', '?')}/-{pr.get('deletions', '?')}).",
        )

        head_sha = pr.get("head_sha")
        if head_sha:
            await client.create_check_run(
                repo,
                name="flyte-triage",
                head_sha=head_sha,
                conclusion="success",
                summary="Flyte triaged this pull request.",
            )
    return f"triaged {repo}#{number}"


if __name__ == "__main__":
    flyte.run(triage_pr, repo="octocat/hello-world", number=1)
