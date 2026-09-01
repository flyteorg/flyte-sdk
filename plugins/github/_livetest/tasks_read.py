"""read_write_pr.py's read path, run against a real repo.

summarize_pr only performs GitHub reads, which succeed unauthenticated against
a public repository, so this exercises the task example end to end without a
valid token.
"""

import flyte

from _livetest.common import REPO, image
from flyteplugins.github import GitHubClient

env = flyte.TaskEnvironment(name="github-read-write", image=image())


@env.task
async def summarize_pr(repo: str, number: int) -> str:
    """Read a pull request and summarize what it changes."""
    async with GitHubClient() as client:
        pr = await client.get_pull_request(repo, number)
        files = await client.get_pull_request_files(repo, number)
    summary = "\n".join(f"- {f['filename']} (+{f['additions']}/-{f['deletions']})" for f in files[:20])
    return f"{pr['title']} ({pr['head']} -> {pr['base']})\n{summary}"


if __name__ == "__main__":
    flyte.init_from_config(project="niels", domain="development")
    run = flyte.with_runcontext(copy_style="all").run(summarize_pr, repo=REPO, number=4)
    print("URL:", run.url)
    run.wait()
    print("OUTPUT:", run.outputs())
