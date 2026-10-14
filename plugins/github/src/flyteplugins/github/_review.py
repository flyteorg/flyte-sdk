"""Pull-request review gates built on `flyte.new_condition`.

The pattern: a task collects review metadata from a pull request, embeds it as
JSON in a markdown condition prompt, parks the run until a human responds in the
Flyte UI, and parses the structured response back into a typed `ReviewDecision`
the workflow can branch on.

This belongs in the plugin rather than in an example because the condition is
the part that only Flyte can do. Reading the pull request is PyGithub's job, and
this module calls it directly rather than wrapping it.

`PyGithub` is an optional extra, so a webhook-only install stays lean:

```bash
pip install "flyteplugins-github[review]"
```
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

#: Environment variable holding the token used to read the pull request.
DEFAULT_TOKEN_ENV_VAR = "GITHUB_TOKEN"

#: Condition names are action names on the backend; keep them comfortably short.
_MAX_CONDITION_NAME = 60

Verdict = Literal["approve", "request_changes", "comment"]


class ReviewComment(BaseModel):
    """A single inline review comment."""

    path: str
    line: int | None = None
    body: str
    severity: Literal["info", "warning", "blocking"] = "info"


class ReviewDecision(BaseModel):
    """Structured decision parsed from a reviewer's condition response."""

    verdict: Verdict
    summary: str = ""
    comments: list[ReviewComment] = Field(default_factory=list)
    reviewer: str | None = None

    @property
    def is_approved(self) -> bool:
        """True when the reviewer approved the change."""
        return self.verdict == "approve"

    @property
    def blocking_comments(self) -> list[ReviewComment]:
        """Comments the reviewer flagged as blocking."""
        return [c for c in self.comments if c.severity == "blocking"]


class ReviewContext(BaseModel):
    """Review metadata collected from a pull request.

    This is the payload embedded in the condition prompt, so the reviewer sees
    everything needed to decide without leaving the Flyte UI.
    """

    repo: str
    number: int
    title: str
    author: str | None = None
    body: str = ""
    base: str | None = None
    head: str | None = None
    url: str | None = None
    additions: int | None = None
    deletions: int | None = None
    changed_files: int | None = None
    files: list[dict[str, Any]] = Field(default_factory=list)
    prior_reviews: list[dict[str, Any]] = Field(default_factory=list)

    def to_json(self, max_file_patches: int = 20) -> str:
        """Serialize to JSON for embedding in a prompt.

        Patches dominate the size of a large diff, so only the first
        `max_file_patches` files keep theirs — the rest keep their stats.
        """
        data = self.model_dump()
        for i, f in enumerate(data["files"]):
            if i >= max_file_patches:
                f.pop("patch", None)
        return json.dumps(data, indent=2)


def _github(token: str | None = None):
    """Build a PyGithub client, with a useful error when the extra is missing."""
    try:
        from github import Auth, Github
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(
            "PyGithub is not installed. Install 'flyteplugins-github[review]' to use the review gate."
        ) from exc

    resolved = token if token is not None else os.environ.get(DEFAULT_TOKEN_ENV_VAR)
    if not resolved:
        raise ValueError(
            f"{DEFAULT_TOKEN_ENV_VAR} is not set. Create the secret and request it on the task's environment: "
            f"secrets=[flyte.Secret('{DEFAULT_TOKEN_ENV_VAR}', as_env_var='{DEFAULT_TOKEN_ENV_VAR}')]"
        )
    return Github(auth=Auth.Token(resolved))


def _collect_sync(repo: str, number: int, max_files: int, token: str | None) -> ReviewContext:
    with _github(token) as gh:
        pull = gh.get_repo(repo).get_pull(number)
        files = [
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            }
            for f in pull.get_files()[:max_files]
        ]
        try:
            prior = [
                {"user": r.user.login if r.user else None, "state": r.state, "body": r.body or ""}
                for r in pull.get_reviews()
            ]
        except Exception:
            # Prior reviews are context, not a requirement — a token without
            # permission to list them should not block the gate.
            prior = []
        return ReviewContext(
            repo=repo,
            number=number,
            title=pull.title,
            author=pull.user.login if pull.user else None,
            body=pull.body or "",
            base=pull.base.ref,
            head=pull.head.ref,
            url=pull.html_url,
            additions=pull.additions,
            deletions=pull.deletions,
            changed_files=pull.changed_files,
            files=files,
            prior_reviews=prior,
        )


async def collect_review_context(
    repo: str,
    number: int,
    *,
    max_files: int = 50,
    token: str | None = None,
) -> ReviewContext:
    """Fetch a pull request and assemble the metadata a reviewer needs.

    Args:
        repo: Repository full name (`owner/repo`).
        number: Pull request number.
        max_files: Cap on how many changed files to include.
        token: Explicit token; otherwise read from `GITHUB_TOKEN`.
    """
    # PyGithub is synchronous, so keep it off the caller's event loop.
    return await asyncio.to_thread(_collect_sync, repo, number, max_files, token)


def build_review_prompt(context: ReviewContext, instructions: str = "") -> str:
    """Build the markdown prompt shown to the reviewer in the Flyte UI.

    The metadata is embedded as a fenced JSON block so it renders verbatim and
    can be machine-parsed downstream.
    """
    instructions = instructions or (
        "Review this pull request. Respond with a JSON object of the form:\n"
        '`{"verdict": "approve" | "request_changes" | "comment", '
        '"summary": "...", "comments": [{"path": "...", "line": 1, '
        '"body": "...", "severity": "info" | "warning" | "blocking"}]}`'
    )
    return (
        f"## Review requested: {context.repo}#{context.number}\n\n"
        f"**{context.title}** (by {context.author or 'unknown'})\n\n"
        f"{context.body}\n\n"
        f"{instructions}\n\n"
        "### Pull request metadata\n\n"
        "```json\n"
        f"{context.to_json()}\n"
        "```\n"
    )


def _normalize_verdict(value: str) -> Verdict:
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    if v in ("approve", "approved", "lgtm", "accept"):
        return "approve"
    if v in ("request_changes", "changes_requested", "reject", "blocked"):
        return "request_changes"
    if v in ("comment", "comments", "neutral", "note"):
        return "comment"
    raise ValueError(f"unknown verdict: {value!r}")


def parse_review_payload(payload: str) -> ReviewDecision:
    """Parse a reviewer's condition response into a `ReviewDecision`.

    Accepts raw JSON, JSON inside a fenced code block, or prose with a JSON
    object somewhere in it — reviewers paste all three. Verdict synonyms
    (`approved`, `changes_requested`, `lgtm`, ...) are normalized.

    Raises:
        ValueError: when no JSON object with a recognizable verdict can be
            extracted from the payload.
    """
    text = (payload or "").strip()
    if not text:
        raise ValueError("empty review payload")

    # Scan every `{` and let raw_decode tolerate trailing content, so a JSON
    # object wrapped in prose or a fenced block is still found.
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "verdict" in obj:
            comments = obj.get("comments") or []
            return ReviewDecision(
                verdict=_normalize_verdict(str(obj["verdict"])),
                summary=str(obj.get("summary") or ""),
                comments=[ReviewComment.model_validate(c) for c in comments if isinstance(c, dict)],
                reviewer=obj.get("reviewer"),
            )
        idx = text.find("{", idx + 1)
    raise ValueError(f"could not extract a review decision from payload: {text[:200]!r}")


def condition_name_for(repo: str, number: int) -> str:
    """Derive a condition name from a pull request, within the length limit."""
    name = f"review-{repo.replace('/', '-')}-{number}"
    if len(name) <= _MAX_CONDITION_NAME:
        return name
    # Keep the number, which is what distinguishes one review from the next.
    suffix = f"-{number}"
    return name[: _MAX_CONDITION_NAME - len(suffix)] + suffix


async def review_pr(
    repo: str,
    number: int,
    *,
    condition_name: str | None = None,
    instructions: str = "",
    timeout: timedelta | int | float | None = None,
    max_files: int = 50,
    token: str | None = None,
) -> ReviewDecision:
    """Park the run on a human review condition and return the decision.

    Collects the pull request's metadata, raises a markdown condition carrying
    it as JSON, waits for a human to respond in the Flyte UI, and parses the
    response into a typed decision:

    ```python
    @env.task
    async def gated_merge(repo: str, number: int) -> str:
        decision = await review_pr(repo, number)
        if decision.is_approved:
            ...
        return f"blocked: {decision.summary}"
    ```

    Args:
        repo: Repository full name (`owner/repo`).
        number: Pull request number.
        condition_name: Name of the condition action; defaults to one derived
            from the repo and number.
        instructions: Override for the reviewer instructions in the prompt.
        timeout: Forwarded to `flyte.new_condition`. On expiry `wait()` raises
            `flyte.errors.ConditionTimedoutError`.
        max_files: Cap on how many changed files to include in the prompt.
        token: Explicit token; otherwise read from `GITHUB_TOKEN`.

    Returns:
        The parsed `ReviewDecision`.

    Raises:
        ValueError: when the reviewer's response carries no recognizable verdict.
    """
    import flyte

    context = await collect_review_context(repo, number, max_files=max_files, token=token)
    condition = await flyte.new_condition.aio(
        condition_name or condition_name_for(repo, number),
        prompt=build_review_prompt(context, instructions=instructions),
        prompt_type="markdown",
        data_type=str,
        timeout=timeout,
    )
    return parse_review_payload(await condition.wait.aio())
