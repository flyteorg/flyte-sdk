"""Pull-request review helpers built on `flyte.new_condition`.

The core pattern: a Flyte task collects review metadata from a PR, embeds it
as JSON in a markdown condition prompt (conditions support `data_type=str`),
parks the run until a human responds in the Flyte UI, and then parses the
structured JSON response back into a typed `ReviewDecision` the rest of the
workflow can branch on.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from ._client import GitHubClient
from ._config import Config, default_config

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

    This is the payload embedded in the condition prompt so the reviewer (or
    an automated reviewer task) sees everything needed to make a decision.
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
        """Serialize the context to a JSON string for embedding in a prompt."""
        data = self.model_dump()
        for i, f in enumerate(data["files"]):
            if i >= max_file_patches:
                f.pop("patch", None)
        return json.dumps(data, indent=2)


async def collect_review_context(
    repo: str,
    number: int,
    *,
    config: Config | None = None,
    max_files: int = 50,
) -> ReviewContext:
    """Fetch a pull request and assemble the metadata a reviewer needs."""
    async with GitHubClient(config or default_config) as client:
        pr = await client.get_pull_request(repo, number)
        files = await client.get_pull_request_files(repo, number, per_page=max_files)
        try:
            reviews = await client.get_pull_request_reviews(repo, number)
        except Exception:
            reviews = []
    return ReviewContext(
        repo=repo,
        number=number,
        title=pr.get("title", ""),
        author=pr.get("user"),
        body=pr.get("body") or "",
        base=pr.get("base"),
        head=pr.get("head"),
        url=pr.get("url"),
        additions=pr.get("additions"),
        deletions=pr.get("deletions"),
        changed_files=pr.get("changed_files"),
        files=files,
        prior_reviews=reviews,
    )


def build_review_prompt(context: ReviewContext, instructions: str = "") -> str:
    """Build the markdown prompt shown to the reviewer in the Flyte UI.

    The PR metadata is embedded as a fenced JSON block so it can be displayed
    verbatim and machine-parsed downstream.
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


def parse_review_payload(payload: str) -> ReviewDecision:
    """Parse a reviewer's condition response into a `ReviewDecision`.

    Accepts raw JSON, JSON inside a fenced code block, or prose containing a
    JSON object anywhere in the text. Verdict synonyms (`approved`,
    `changes_requested`, `lgtm`, ...) are normalized.

    Raises:
        ValueError: when no JSON object with a recognizable verdict can be
            extracted from the payload.
    """
    text = (payload or "").strip()
    if not text:
        raise ValueError("empty review payload")

    # Collect every JSON object embedded in the text (raw JSON, fenced code
    # blocks, or prose) using raw_decode, which tolerates trailing content.
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                candidates.append(obj)
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)

    for data in candidates:
        if "verdict" not in data:
            continue
        verdict = _normalize_verdict(str(data["verdict"]))
        comments = data.get("comments") or []
        return ReviewDecision(
            verdict=verdict,
            summary=str(data.get("summary") or ""),
            comments=[ReviewComment.model_validate(c) for c in comments if isinstance(c, dict)],
            reviewer=data.get("reviewer"),
        )
    raise ValueError(f"could not extract a review decision from payload: {text[:200]!r}")


def _normalize_verdict(value: str) -> Verdict:
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    if v in ("approve", "approved", "lgtm", "accept"):
        return "approve"
    if v in ("request_changes", "changes_requested", "reject", "blocked"):
        return "request_changes"
    if v in ("comment", "comments", "neutral", "note"):
        return "comment"
    raise ValueError(f"unknown verdict: {value!r}")


async def review_pr(
    repo: str,
    number: int,
    *,
    condition_name: str | None = None,
    instructions: str = "",
    timeout: timedelta | int | float | None = None,
    config: Config | None = None,
) -> ReviewDecision:
    """Park the run on a human review condition and return the decision.

    This is the end-to-end helper: collect PR metadata, create a markdown
    condition whose payload carries the metadata as JSON, wait for the human
    response, and parse it into a `ReviewDecision`.

    Args:
        repo: Repository full name (`owner/repo`).
        number: Pull request number.
        condition_name: Name of the condition action; defaults to
            `review-<owner>-<repo>-<number>`.
        instructions: Optional override for the reviewer instructions.
        timeout: Optional condition timeout, forwarded to `flyte.new_condition`.
        config: Optional plugin configuration.

    Returns:
        The parsed `ReviewDecision`.
    """
    import flyte

    context = await collect_review_context(repo, number, config=config)
    name = condition_name or f"review-{repo.replace('/', '-')}-{number}"
    condition = await flyte.new_condition.aio(
        name,
        prompt=build_review_prompt(context, instructions=instructions),
        prompt_type="markdown",
        data_type=str,
        timeout=timeout,
    )
    payload = await condition.wait.aio()
    return parse_review_payload(payload)
