"""Tests for the GitHub API client."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from flyteplugins.github import GitHubAPIError, GitHubClient, MissingCredentialsError


async def test_get_pull_request(github_api):
    github_api.get("/repos/octo/repo/pulls/42").respond(
        json={
            "number": 42,
            "title": "Add feature",
            "state": "open",
            "body": "A description",
            "user": {"login": "octocat"},
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "draft": False,
            "merged": False,
            "additions": 10,
            "deletions": 2,
            "changed_files": 3,
            "html_url": "https://github.com/octo/repo/pull/42",
            "labels": [{"name": "enhancement"}],
        }
    )
    async with GitHubClient(token="t") as client:
        pr = await client.get_pull_request("octo/repo", 42)
    assert pr["number"] == 42
    assert pr["head"] == "feature"
    assert pr["labels"] == ["enhancement"]
    assert pr["user"] == "octocat"


async def test_list_issues_excludes_pull_requests(github_api):
    github_api.get("/repos/octo/repo/issues").respond(
        json=[
            {"number": 1, "title": "Bug", "state": "open", "labels": []},
            {"number": 2, "title": "A PR", "state": "open", "labels": [], "pull_request": {"url": "..."}},
        ]
    )
    async with GitHubClient(token="t") as client:
        issues = await client.list_issues("octo/repo")
    assert [i["number"] for i in issues] == [1]
    assert issues[0]["is_pull_request"] is False


async def test_get_file_contents_decodes_base64(github_api):
    content = base64.b64encode(b"hello world").decode()
    github_api.get("/repos/octo/repo/contents/README.md").respond(json={"content": content, "encoding": "base64"})
    async with GitHubClient(token="t") as client:
        text = await client.get_file_contents("octo/repo", "README.md")
    assert text == "hello world"


async def test_list_repository_files_filters_blobs(github_api):
    github_api.get("/repos/octo/repo/git/trees/main").respond(
        json={
            "tree": [
                {"path": "src", "type": "tree", "sha": "a"},
                {"path": "src/app.py", "type": "blob", "sha": "b", "size": 10},
                {"path": "README.md", "type": "blob", "sha": "c", "size": 5},
            ]
        }
    )
    async with GitHubClient(token="t") as client:
        files = await client.list_repository_files("octo/repo", ref="main", path="src/")
    assert files == [{"path": "src/app.py", "size": 10, "sha": "b"}]


async def test_create_issue_requires_auth(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    async with GitHubClient(token=None) as client:
        with pytest.raises(MissingCredentialsError) as excinfo:
            await client.create_issue("octo/repo", "title")
    assert "GITHUB_TOKEN" in str(excinfo.value)


async def test_api_error_includes_message(github_api):
    github_api.get("/repos/octo/repo/pulls/1").respond(status_code=404, json={"message": "Not Found"})
    async with GitHubClient(token="t") as client:
        with pytest.raises(GitHubAPIError) as excinfo:
            await client.get_pull_request("octo/repo", 1)
    assert excinfo.value.status_code == 404
    assert "Not Found" in str(excinfo.value)


async def test_retries_on_500(github_api):
    route = github_api.get("/repos/octo/repo")
    route.side_effect = [
        httpx.Response(500, json={"message": "boom"}),
        httpx.Response(200, json={"full_name": "octo/repo"}),
    ]
    from flyteplugins.github import Config

    config = Config(retry_backoff=0.0)
    async with GitHubClient(config, token="t") as client:
        repo = await client.get_repository("octo/repo")
    assert repo["full_name"] == "octo/repo"
    assert route.call_count == 2


async def test_create_pull_request_review_payload(github_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"id": 1, "state": "APPROVED", "user": {"login": "reviewer"}})

    github_api.post("/repos/octo/repo/pulls/42/reviews").mock(side_effect=capture)
    async with GitHubClient(token="t") as client:
        review = await client.create_pull_request_review(
            "octo/repo", 42, "APPROVE", body="lgtm", comments=[{"path": "a.py", "line": 3, "body": "nice"}]
        )
    assert review["state"] == "APPROVED"
    assert captured["body"]["event"] == "APPROVE"
    assert captured["body"]["comments"][0]["path"] == "a.py"


async def test_create_or_update_file_fetches_existing_sha(github_api):
    github_api.get("/repos/octo/repo/contents/docs.md").respond(json={"sha": "abc123"})

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        assert body["sha"] == "abc123"
        assert base64.b64decode(body["content"]) == b"new content"
        return httpx.Response(200, json={"commit": {"sha": "def456"}})

    github_api.put("/repos/octo/repo/contents/docs.md").mock(side_effect=capture)
    async with GitHubClient(token="t") as client:
        result = await client.create_or_update_file("octo/repo", "docs.md", "new content", "update docs")
    assert result["sha"] == "def456"


async def test_create_or_update_file_new_file(github_api):
    github_api.get("/repos/octo/repo/contents/new.md").respond(status_code=404, json={"message": "Not Found"})
    github_api.put("/repos/octo/repo/contents/new.md").respond(json={"commit": {"sha": "s1"}})
    async with GitHubClient(token="t") as client:
        result = await client.create_or_update_file("octo/repo", "new.md", "x", "add")
    assert result["sha"] == "s1"


async def test_create_branch(github_api):
    github_api.get("/repos/octo/repo/git/ref/heads/main").respond(json={"object": {"sha": "sha-main"}})
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(201, json={})

    github_api.post("/repos/octo/repo/git/refs").mock(side_effect=capture)
    async with GitHubClient(token="t") as client:
        sha = await client.create_branch("octo/repo", "agent/fix", from_ref="main")
    assert sha == "sha-main"
    assert captured["body"] == {"ref": "refs/heads/agent/fix", "sha": "sha-main"}


async def test_merge_pull_request(github_api):
    github_api.put("/repos/octo/repo/pulls/42/merge").respond(json={"merged": True, "sha": "m1"})
    async with GitHubClient(token="t") as client:
        result = await client.merge_pull_request("octo/repo", 42, merge_method="squash")
    assert result["merged"] is True


async def test_client_requires_context_manager():
    client = GitHubClient(token="t")
    with pytest.raises(RuntimeError):
        await client.get_user()


async def test_create_branch_resolves_the_repo_default_branch(github_api):
    """`from_ref="HEAD"` must follow the repo's default branch, not assume `main`."""
    github_api.get("/repos/octo/repo").respond(json={"default_branch": "trunk"})
    ref_route = github_api.get("/repos/octo/repo/git/ref/heads/trunk").respond(json={"object": {"sha": "abc123"}})
    create = github_api.post("/repos/octo/repo/git/refs").respond(json={})
    async with GitHubClient(token="t") as client:
        sha = await client.create_branch("octo/repo", "feature")
    assert sha == "abc123"
    assert ref_route.called
    assert json.loads(create.calls[0].request.content) == {"ref": "refs/heads/feature", "sha": "abc123"}


async def test_create_or_update_file_reads_the_existing_sha_from_the_target_branch(github_api):
    """Without `ref`, GitHub answers from the default branch and the SHA is wrong."""
    get_route = github_api.get("/repos/octo/repo/contents/docs/x.md").respond(json={"sha": "branch-sha"})
    put_route = github_api.put("/repos/octo/repo/contents/docs/x.md").respond(json={"commit": {"sha": "new"}})
    async with GitHubClient(token="t") as client:
        await client.create_or_update_file("octo/repo", "docs/x.md", "body", "msg", branch="feature")
    assert get_route.calls[0].request.url.params["ref"] == "feature"
    assert json.loads(put_route.calls[0].request.content)["sha"] == "branch-sha"


async def test_request_retries_a_secondary_rate_limit(github_api):
    github_api.get("/repos/octo/repo").mock(
        side_effect=[
            httpx.Response(403, headers={"retry-after": "0"}, json={"message": "secondary rate limit"}),
            httpx.Response(200, json={"full_name": "octo/repo"}),
        ]
    )
    async with GitHubClient(token="t") as client:
        repo = await client.get_repository("octo/repo")
    assert repo["full_name"] == "octo/repo"


async def test_request_does_not_retry_a_plain_403(github_api):
    """A permissions 403 is not a rate limit and must surface immediately."""
    route = github_api.get("/repos/octo/repo").respond(403, json={"message": "Resource not accessible"})
    async with GitHubClient(token="t") as client:
        with pytest.raises(GitHubAPIError) as exc:
            await client.get_repository("octo/repo")
    assert exc.value.status_code == 403
    assert len(route.calls) == 1
