"""Human-gated PR merging: a condition carrying a JSON payload.

A task collects review metadata from a pull request, embeds it as JSON in a
markdown condition prompt, parks the run until a human answers in the Flyte UI,
and parses the response into a typed decision. Approved PRs get merged.

The gate itself is `flyteplugins.github.review_pr` — it lives in the plugin
because `flyte.new_condition` is the part only Flyte can do. Merging is
`PyGithub`, called directly here.

Requirements:
    pip install "flyteplugins-github[review]"

Setup:
    flyte create secret GITHUB_TOKEN --value <token-with-repo-scope>

Usage:
    flyte run examples/external_saas_integrations/github_pr_review_gate.py \\
        gated_merge --repo octocat/hello-world --number 1
"""

import asyncio
import os

from flyteplugins.github import ReviewDecision, review_pr

import flyte

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-github[review]")

env = flyte.TaskEnvironment(
    name="github-review-gate",
    image=image,
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
)


def _comment(repo: str, number: int, body: str) -> None:
    from github import Auth, Github

    with Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])) as gh:
        gh.get_repo(repo).get_issue(number).create_comment(body)


def _merge(repo: str, number: int) -> str:
    from github import Auth, Github

    with Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])) as gh:
        result = gh.get_repo(repo).get_pull(number).merge(merge_method="squash")
    return f"merged {result.sha}"


@env.task
async def gated_merge(repo: str, number: int) -> str:
    """Wait for a human review, then merge if approved.

    The run parks at `review_pr` until someone answers the condition in the
    Flyte UI. Pass `timeout=` to bound that wait.
    """
    decision: ReviewDecision = await review_pr(repo, number)

    if not decision.is_approved:
        # Post the reviewer's reasoning back to the PR before bailing out, so
        # the decision is visible where the author is looking.
        blockers = "\n".join(f"- `{c.path}`: {c.body}" for c in decision.blocking_comments)
        await asyncio.to_thread(
            _comment, repo, number, f"Review gate blocked this merge: {decision.summary}\n{blockers}"
        )
        return f"blocked: {decision.summary}"

    # PyGithub is synchronous; keep it off the event loop.
    return await asyncio.to_thread(_merge, repo, number)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext().run(gated_merge, repo="octocat/hello-world", number=1)
    print(run.url)
