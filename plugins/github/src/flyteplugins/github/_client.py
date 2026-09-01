"""Async GitHub REST API client used by tasks, webhooks, and the MCP server.

The client is intentionally thin: one `httpx.AsyncClient` per request context,
retry with exponential backoff on transient failures, and typed errors for API
failures. Every method corresponds to a single GitHub REST endpoint so the
surface is easy to reason about from Flyte tasks.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any

import httpx
from flyte.syncify import syncify

from ._config import Config, default_config
from ._errors import GitHubAPIError, MissingCredentialsError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {500, 502, 503, 504}

#: Never sleep longer than this on a rate-limit retry. A reset window further out
#: is better surfaced as an error than silently held inside a task.
MAX_RATE_LIMIT_SLEEP = 60.0


def _rate_limit_delay(response: httpx.Response) -> float | None:
    """Seconds to wait before retrying a rate-limited response, or None.

    GitHub signals rate limiting either as 429, or as 403 carrying a
    `Retry-After` header (secondary limits) or an exhausted quota with an
    `x-ratelimit-reset` epoch (primary limits). Returns None when the response
    is not rate limiting, when the header is unparsable, or when the reset is
    further out than `MAX_RATE_LIMIT_SLEEP` — waiting an hour inside a request
    is worse than raising.
    """
    if response.status_code not in (403, 429):
        return None
    headers = response.headers
    raw_retry_after = headers.get("retry-after")
    if raw_retry_after:
        try:
            return min(max(float(raw_retry_after), 0.0), MAX_RATE_LIMIT_SLEEP)
        except ValueError:
            return None
    if headers.get("x-ratelimit-remaining") == "0":
        try:
            delay = float(headers["x-ratelimit-reset"]) - time.time()
        except (KeyError, ValueError):
            return None
        if delay <= 0:
            return 0.0
        return delay if delay <= MAX_RATE_LIMIT_SLEEP else None
    return None


class GitHubClient:
    """Async client for the GitHub REST API.

    Use as an async context manager:

    ```python
    from flyteplugins.github import GitHubClient

    async with GitHubClient() as client:
        pr = await client.get_pull_request("octocat/hello-world", 42)
    ```

    Args:
        config: Plugin configuration. Defaults to the module-level
            `default_config`.
        token: Explicit token. When omitted, the token is read from the
            environment variable named by `config.token_env`. Read-only calls
            work without a token (subject to GitHub's anonymous rate limits);
            write calls raise `MissingCredentialsError` when no token is found.
    """

    def __init__(self, config: Config | None = None, token: str | None = None):
        self.config = config or default_config
        self._token = token
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> GitHubClient:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.config.user_agent,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = self._token if self._token is not None else self.config.token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            headers=headers,
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def __enter__(self) -> GitHubClient:
        """Enter synchronously, for use with the blocking call form.

        `__aenter__` runs on syncify's background loop — the same loop the
        syncified methods run on — so the underlying `httpx.AsyncClient` is
        created and used on a single loop.
        """
        return self._enter_sync()

    def __exit__(self, *exc_info: object) -> None:
        self._exit_sync()

    @syncify
    async def _enter_sync(self) -> GitHubClient:
        return await self.__aenter__()

    @syncify
    async def _exit_sync(self) -> None:
        await self.__aexit__()

    # ------------------------------------------------------------------
    # low-level request
    # ------------------------------------------------------------------

    @syncify
    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        require_auth: bool = False,
    ) -> Any:
        """Send a request to the GitHub API, retrying transient failures.

        Args:
            method: HTTP method.
            path: URL path relative to the API base URL (e.g. `/repos/o/r`).
            params: Optional query parameters.
            json: Optional JSON request body.
            require_auth: When True, raise `MissingCredentialsError` before
                sending if no token is configured.

        Returns:
            Parsed JSON response body, or `None` for 204 responses.

        Raises:
            MissingCredentialsError: if `require_auth` is set and no token is
                mounted.
            GitHubAPIError: on any non-2xx response after retries.
        """
        if self._client is None:
            raise RuntimeError("GitHubClient must be used as an async context manager (async with ...).")
        if require_auth and "Authorization" not in self._client.headers:
            raise MissingCredentialsError(self.config.token_env)

        backoff = self.config.retry_backoff
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, path, params=params, json=json)
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise GitHubAPIError(0, f"transport error: {exc}", url=str(self._client.base_url) + path) from exc
                logger.warning("GitHub transport error (%s), retrying in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code < 400:
                if response.status_code == 204 or not response.content:
                    return None
                return response.json()

            rate_limit_delay = _rate_limit_delay(response)
            if rate_limit_delay is not None and attempt < self.config.max_retries:
                logger.warning("GitHub rate limited, retrying in %.1fs", rate_limit_delay)
                await asyncio.sleep(rate_limit_delay)
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                logger.warning("GitHub returned %s, retrying in %.1fs", response.status_code, backoff)
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            message = _error_message(response)
            raise GitHubAPIError(response.status_code, message, url=str(response.url), body=_safe_json(response))

    # ------------------------------------------------------------------
    # read: repositories, users
    # ------------------------------------------------------------------

    @syncify
    async def get_user(self) -> dict[str, Any]:
        """Return the authenticated user's profile (`GET /user`)."""
        return await self.request.aio("GET", "/user", require_auth=True)

    @syncify
    async def get_repository(self, repo: str) -> dict[str, Any]:
        """Return metadata for a repository (`GET /repos/{repo}`)."""
        return await self.request.aio("GET", f"/repos/{repo}")

    @syncify
    async def list_repositories(self, org: str | None = None, per_page: int = 30) -> list[dict[str, Any]]:
        """List repositories for the authenticated user or an organization."""
        path = f"/orgs/{org}/repos" if org else "/user/repos"
        return await self.request.aio("GET", path, params={"per_page": per_page})

    @syncify
    async def list_repository_files(self, repo: str, ref: str | None = None, path: str = "") -> list[dict[str, Any]]:
        """List file paths in a repository tree via the git trees API.

        Returns a list of `{"path", "size", "sha"}` dicts for every blob
        (file) under `path`.
        """
        # The git trees API takes the ref in the path; it has no `ref` query parameter.
        tree = await self.request.aio("GET", f"/repos/{repo}/git/trees/{ref or 'HEAD'}", params={"recursive": "1"})
        return [
            {"path": entry["path"], "size": entry.get("size"), "sha": entry["sha"]}
            for entry in tree.get("tree", [])
            if entry.get("type") == "blob" and entry["path"].startswith(path)
        ]

    @syncify
    async def get_file_contents(self, repo: str, path: str, ref: str | None = None) -> str:
        """Read a file from a repository and return its decoded text content."""
        params = {"ref": ref} if ref else None
        data = await self.request.aio("GET", f"/repos/{repo}/contents/{path}", params=params)
        if isinstance(data, list):
            raise GitHubAPIError(200, f"{path} is a directory; pass a file path", url=path)
        if data.get("encoding") == "base64":
            return base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        return data.get("content", "")

    @syncify
    async def list_commits(
        self, repo: str, sha: str | None = None, path: str | None = None, per_page: int = 30
    ) -> list[dict[str, Any]]:
        """List commits on a repository, optionally filtered by branch or path."""
        params: dict[str, Any] = {"per_page": per_page}
        if sha:
            params["sha"] = sha
        if path:
            params["path"] = path
        commits = await self.request.aio("GET", f"/repos/{repo}/commits", params=params)
        return [
            {
                "sha": c["sha"],
                "message": c["commit"]["message"],
                "author": (c["commit"].get("author") or {}).get("name"),
                "date": (c["commit"].get("author") or {}).get("date"),
                "url": c.get("html_url"),
            }
            for c in commits
        ]

    # ------------------------------------------------------------------
    # read: issues and pull requests
    # ------------------------------------------------------------------

    @syncify
    async def list_issues(self, repo: str, state: str = "open", per_page: int = 30) -> list[dict[str, Any]]:
        """List issues (excluding pull requests) on a repository."""
        issues = await self.request.aio("GET", f"/repos/{repo}/issues", params={"state": state, "per_page": per_page})
        return [_simplify_issue(i) for i in issues if "pull_request" not in i]

    @syncify
    async def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        """Return a single issue or pull request by number."""
        issue = await self.request.aio("GET", f"/repos/{repo}/issues/{number}")
        return _simplify_issue(issue)

    @syncify
    async def list_issue_comments(self, repo: str, number: int, per_page: int = 30) -> list[dict[str, Any]]:
        """List comments on an issue or pull request."""
        comments = await self.request.aio(
            "GET", f"/repos/{repo}/issues/{number}/comments", params={"per_page": per_page}
        )
        return [
            {
                "id": c["id"],
                "user": (c.get("user") or {}).get("login"),
                "body": c.get("body") or "",
                "created_at": c.get("created_at"),
            }
            for c in comments
        ]

    @syncify
    async def list_pull_requests(self, repo: str, state: str = "open", per_page: int = 30) -> list[dict[str, Any]]:
        """List pull requests on a repository."""
        prs = await self.request.aio("GET", f"/repos/{repo}/pulls", params={"state": state, "per_page": per_page})
        return [_simplify_pr(p) for p in prs]

    @syncify
    async def get_pull_request(self, repo: str, number: int) -> dict[str, Any]:
        """Return a single pull request by number."""
        pr = await self.request.aio("GET", f"/repos/{repo}/pulls/{number}")
        return _simplify_pr(pr)

    @syncify
    async def get_pull_request_files(self, repo: str, number: int, per_page: int = 100) -> list[dict[str, Any]]:
        """List files changed by a pull request, with per-file diff stats."""
        files = await self.request.aio("GET", f"/repos/{repo}/pulls/{number}/files", params={"per_page": per_page})
        return [
            {
                "filename": f["filename"],
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                "changes": f.get("changes"),
                "patch": f.get("patch"),
            }
            for f in files
        ]

    @syncify
    async def get_pull_request_reviews(self, repo: str, number: int) -> list[dict[str, Any]]:
        """List reviews submitted on a pull request."""
        reviews = await self.request.aio("GET", f"/repos/{repo}/pulls/{number}/reviews")
        return [
            {
                "id": r["id"],
                "user": (r.get("user") or {}).get("login"),
                "state": r.get("state"),
                "body": r.get("body") or "",
                "submitted_at": r.get("submitted_at"),
            }
            for r in reviews
        ]

    # ------------------------------------------------------------------
    # write: issues
    # ------------------------------------------------------------------

    @syncify
    async def create_issue(
        self,
        repo: str,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an issue on a repository."""
        payload: dict[str, Any] = {"title": title}
        if body is not None:
            payload["body"] = body
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        issue = await self.request.aio("POST", f"/repos/{repo}/issues", json=payload, require_auth=True)
        return _simplify_issue(issue)

    @syncify
    async def create_issue_comment(self, repo: str, number: int, body: str) -> dict[str, Any]:
        """Comment on an issue or pull request."""
        comment = await self.request.aio(
            "POST", f"/repos/{repo}/issues/{number}/comments", json={"body": body}, require_auth=True
        )
        return {
            "id": comment["id"],
            "user": (comment.get("user") or {}).get("login"),
            "body": comment.get("body") or "",
            "url": comment.get("html_url"),
        }

    @syncify
    async def update_issue(
        self,
        repo: str,
        number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an issue's title, body, state, or labels."""
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if state is not None:
            payload["state"] = state
        if labels is not None:
            payload["labels"] = labels
        issue = await self.request.aio("PATCH", f"/repos/{repo}/issues/{number}", json=payload, require_auth=True)
        return _simplify_issue(issue)

    @syncify
    async def add_labels(self, repo: str, number: int, labels: list[str]) -> list[str]:
        """Add labels to an issue or pull request; returns the applied label names."""
        applied = await self.request.aio(
            "POST", f"/repos/{repo}/issues/{number}/labels", json={"labels": labels}, require_auth=True
        )
        return [label["name"] for label in applied]

    # ------------------------------------------------------------------
    # write: pull requests
    # ------------------------------------------------------------------

    @syncify
    async def create_pull_request(
        self,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        """Open a pull request from `head` into `base`."""
        payload: dict[str, Any] = {"title": title, "head": head, "base": base, "draft": draft}
        if body is not None:
            payload["body"] = body
        pr = await self.request.aio("POST", f"/repos/{repo}/pulls", json=payload, require_auth=True)
        return _simplify_pr(pr)

    @syncify
    async def update_pull_request(
        self,
        repo: str,
        number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Update a pull request's title, body, or open/closed state."""
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if state is not None:
            payload["state"] = state
        pr = await self.request.aio("PATCH", f"/repos/{repo}/pulls/{number}", json=payload, require_auth=True)
        return _simplify_pr(pr)

    @syncify
    async def create_pull_request_review(
        self,
        repo: str,
        number: int,
        event: str,
        body: str | None = None,
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Submit a review on a pull request.

        Args:
            repo: Repository full name (`owner/repo`).
            number: Pull request number.
            event: One of `APPROVE`, `REQUEST_CHANGES`, or `COMMENT`.
            body: Review summary text.
            comments: Optional inline comments, each a dict with `path`,
                `line` (or `position`), and `body` keys.
        """
        payload: dict[str, Any] = {"event": event}
        if body is not None:
            payload["body"] = body
        if comments:
            payload["comments"] = comments
        review = await self.request.aio(
            "POST", f"/repos/{repo}/pulls/{number}/reviews", json=payload, require_auth=True
        )
        return {
            "id": review["id"],
            "state": review.get("state"),
            "user": (review.get("user") or {}).get("login"),
            "url": review.get("html_url"),
        }

    @syncify
    async def merge_pull_request(
        self,
        repo: str,
        number: int,
        merge_method: str = "merge",
        commit_title: str | None = None,
        commit_message: str | None = None,
    ) -> dict[str, Any]:
        """Merge a pull request using `merge`, `squash`, or `rebase`."""
        payload: dict[str, Any] = {"merge_method": merge_method}
        if commit_title is not None:
            payload["commit_title"] = commit_title
        if commit_message is not None:
            payload["commit_message"] = commit_message
        return await self.request.aio("PUT", f"/repos/{repo}/pulls/{number}/merge", json=payload, require_auth=True)

    # ------------------------------------------------------------------
    # write: refs, files, checks
    # ------------------------------------------------------------------

    @syncify
    async def create_branch(self, repo: str, branch: str, from_ref: str = "HEAD") -> str:
        """Create a branch at the given ref and return the new branch's SHA.

        `from_ref="HEAD"` means the repository's own default branch, which is
        resolved from the repo metadata — it is not always `main`.
        """
        if from_ref in ("HEAD", "head"):
            repo_data = await self.request.aio("GET", f"/repos/{repo}")
            from_ref = repo_data.get("default_branch") or "main"
        ref_data = await self.request.aio("GET", f"/repos/{repo}/git/ref/{_ref_path(from_ref)}", require_auth=True)
        sha = ref_data["object"]["sha"]
        await self.request.aio(
            "POST",
            f"/repos/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": sha},
            require_auth=True,
        )
        return sha

    @syncify
    async def create_or_update_file(
        self,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str | None = None,
    ) -> dict[str, Any]:
        """Create or update a single file via the contents API.

        Fetches the current blob SHA when the file exists (required by GitHub
        for updates), then commits the new content.
        """
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        }
        if branch:
            payload["branch"] = branch
        try:
            # Read the current blob from the branch we are about to write to. Without
            # `ref`, GitHub answers from the default branch and the returned SHA either
            # fails the update or resurrects default-branch content on the target branch.
            existing = await self.request.aio(
                "GET", f"/repos/{repo}/contents/{path}", params={"ref": branch} if branch else None
            )
            if isinstance(existing, dict) and existing.get("sha"):
                payload["sha"] = existing["sha"]
        except GitHubAPIError as exc:
            if exc.status_code != 404:
                raise
        result = await self.request.aio("PUT", f"/repos/{repo}/contents/{path}", json=payload, require_auth=True)
        commit = result.get("commit") or {}
        return {"sha": commit.get("sha"), "path": path, "branch": branch}

    @syncify
    async def create_check_run(
        self,
        repo: str,
        name: str,
        head_sha: str,
        status: str = "completed",
        conclusion: str | None = None,
        title: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Create a check run reporting the result of an external process.

        Args:
            status: `queued`, `in_progress`, or `completed`.
            conclusion: Required when status is `completed`; one of `success`,
                `failure`, `neutral`, `cancelled`, `skipped`, `action_required`.
        """
        payload: dict[str, Any] = {"name": name, "head_sha": head_sha, "status": status}
        if conclusion is not None:
            payload["conclusion"] = conclusion
        if title or summary:
            payload["output"] = {"title": title or name, "summary": summary or ""}
        check = await self.request.aio("POST", f"/repos/{repo}/check-runs", json=payload, require_auth=True)
        return {
            "id": check["id"],
            "name": check.get("name"),
            "status": check.get("status"),
            "url": check.get("html_url"),
        }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _ref_path(ref: str) -> str:
    if ref.startswith("refs/"):
        return ref
    if ref.startswith(("heads/", "tags/")):
        return ref
    return f"heads/{ref}"


def _safe_json(response: httpx.Response) -> dict[str, Any] | None:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return None


def _error_message(response: httpx.Response) -> str:
    body = _safe_json(response)
    if body:
        if isinstance(body.get("message"), str):
            errors = body.get("errors")
            if isinstance(errors, list) and errors:
                details = "; ".join(e.get("message", str(e)) for e in errors if isinstance(e, dict))
                return f"{body['message']}: {details}"
            return str(body["message"])
    return response.text[:300] or f"HTTP {response.status_code}"


def _simplify_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "state": issue.get("state"),
        "body": issue.get("body") or "",
        "user": (issue.get("user") or {}).get("login"),
        "labels": [label["name"] for label in issue.get("labels", [])],
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "url": issue.get("html_url"),
        "is_pull_request": "pull_request" in issue,
    }


def _simplify_pr(pr: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "state": pr.get("state"),
        "body": pr.get("body") or "",
        "user": (pr.get("user") or {}).get("login"),
        "head": (pr.get("head") or {}).get("ref"),
        "head_sha": (pr.get("head") or {}).get("sha"),
        "base": (pr.get("base") or {}).get("ref"),
        "draft": pr.get("draft", False),
        "merged": pr.get("merged", False),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changed_files"),
        "created_at": pr.get("created_at"),
        "url": pr.get("html_url"),
        "labels": [label["name"] for label in pr.get("labels", [])],
    }
