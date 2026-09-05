"""Read and write GitHub from tasks, with PyGithub.

The task the webhook receiver launches when a PR opens. There is no Flyte
plugin between you and GitHub here — PyGithub is the maintained client, and a
task is just a function that calls it.

Requirements:
    pip install flyte PyGithub

Setup:
    flyte create secret GITHUB_TOKEN --value <token-with-repo-scope>

Usage:
    flyte run examples/external_saas_integrations/github_triage_pr.py \\
        triage_pr --repo octocat/hello-world --number 1
"""

import os

import flyte

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("PyGithub")

env = flyte.TaskEnvironment(
    name="github-triage",
    image=image,
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
)


def _client():
    from github import Auth, Github

    return Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))


@env.task
def summarize_pr(repo: str, number: int) -> str:
    """Read a pull request and summarize what it changes."""
    with _client() as gh:
        pr = gh.get_repo(repo).get_pull(number)
        lines = [f"- {f.filename} (+{f.additions}/-{f.deletions})" for f in pr.get_files()[:20]]
        return f"{pr.title} ({pr.head.ref} -> {pr.base.ref})\n" + "\n".join(lines)


@env.task
def triage_pr(repo: str, number: int) -> str:
    """Label a new PR, comment on it, and report a check run.

    This is what `webhook_receiver.py` launches for every newly opened PR.
    """
    with _client() as gh:
        repository = gh.get_repo(repo)
        pr = repository.get_pull(number)
        pr.add_to_labels("flyte-triage")
        pr.create_issue_comment(
            f"Flyte triage: this PR touches {pr.changed_files} files (+{pr.additions}/-{pr.deletions})."
        )
        repository.create_check_run(
            name="flyte-triage",
            head_sha=pr.head.sha,
            status="completed",
            conclusion="success",
            output={"title": "flyte-triage", "summary": "Flyte triaged this pull request."},
        )
    return f"triaged {repo}#{number}"


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(triage_pr, repo="octocat/hello-world", number=1).url)
