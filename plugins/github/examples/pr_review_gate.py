"""Human-gated PR merging with a flyte condition carrying a JSON payload.

This example shows the plugin's headline pattern: a task collects review
metadata from a GitHub pull request, embeds it as JSON in a markdown
condition prompt, parks the run until a human responds in the Flyte UI, and
parses the structured JSON response into a `ReviewDecision` the workflow can
branch on. When approved, the PR is merged.

Requirements:
    pip install flyteplugins-github

Setup:
    flyte create secret GITHUB_TOKEN --value <token-with-repo-scope>

Usage:
    python plugins/github/examples/pr_review_gate.py
"""

import flyte

from flyteplugins.github import GitHubClient, review_pr

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-github")

env = flyte.TaskEnvironment(
    name="github-review-gate",
    image=image,
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
)


@env.task
async def gated_merge(repo: str, number: int) -> str:
    """Wait for a human PR review, then merge if approved."""
    decision = await review_pr(repo, number)

    if not decision.is_approved:
        # Post the reviewer's feedback back to the PR before bailing out.
        async with GitHubClient() as client:
            await client.create_issue_comment.aio(
                repo,
                number,
                f"Review gate blocked this merge: {decision.summary}",
            )
        return f"blocked: {decision.summary}"

    async with GitHubClient() as client:
        result = await client.merge_pull_request.aio(repo, number, merge_method="squash")
    return f"merged {result.get('sha', '')}"


if __name__ == "__main__":
    # For a real run, replace with a repo/PR you control. Locally you can run:
    #   flyte run plugins/github/examples/pr_review_gate.py --repo octocat/hello-world --number 1
    flyte.init_from_config()
    run = flyte.with_runcontext().run(gated_merge, repo="octocat/hello-world", number=1)
    print(run.url)
