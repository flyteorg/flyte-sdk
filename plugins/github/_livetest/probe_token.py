"""Verify the GITHUB_PAT cluster secret works as GITHUB_TOKEN."""

import flyte

from _livetest.common import GH_SECRET, REPO, image

env = flyte.TaskEnvironment(name="gh-probe", image=image(), secrets=[GH_SECRET])


@env.task
async def probe() -> str:
    import os

    from flyteplugins.github import GitHubClient

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "FAIL: GITHUB_TOKEN not mounted"
    async with GitHubClient() as c:
        me = await c.get_user()
        repo = await c.get_repository(REPO)
        prs = await c.list_pull_requests(REPO, state="open")
    return (
        f"OK login={me.get('login')} token_prefix={token[:4]} "
        f"repo={repo.get('full_name')} open_prs={[p.get('number') for p in prs]}"
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(copy_style="all").run(probe)
    print("URL:", run.url)
    run.wait()
    print("OUTPUT:", run.outputs())
