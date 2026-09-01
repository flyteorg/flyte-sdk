"""Human-gated PR merging: a condition carrying a JSON payload.

The headline pattern. A task collects review metadata from a pull request,
embeds it as JSON in a markdown condition prompt, parks the run until a human
responds in the Flyte UI, and parses the structured response into a typed
decision the workflow branches on. Approved PRs get merged.

Conditions are Flyte's; talking to GitHub is PyGithub's. Nothing here wraps the
GitHub API — `flyte.new_condition` is the only part that needed inventing.

Requirements:
    pip install flyte PyGithub

Setup:
    flyte create secret GITHUB_TOKEN --value <token-with-repo-scope>

Usage:
    flyte run examples/external_saas_integrations/github_pr_review_gate.py \\
        gated_merge --repo octocat/hello-world --number 1
"""

import asyncio
import json
import os
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

import flyte

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("PyGithub", "pydantic")

env = flyte.TaskEnvironment(
    name="github-review-gate",
    image=image,
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
)

Verdict = Literal["approve", "request_changes", "comment"]


class ReviewComment(BaseModel):
    """A single inline review comment."""

    path: str
    line: int | None = None
    body: str
    severity: Literal["info", "warning", "blocking"] = "info"


class ReviewDecision(BaseModel):
    """The reviewer's answer, parsed out of their condition response."""

    verdict: Verdict
    summary: str = ""
    comments: list[ReviewComment] = Field(default_factory=list)

    @property
    def is_approved(self) -> bool:
        return self.verdict == "approve"

    @property
    def blocking_comments(self) -> list[ReviewComment]:
        return [c for c in self.comments if c.severity == "blocking"]


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
    """Parse a reviewer's response into a `ReviewDecision`.

    Accepts raw JSON, JSON inside a fenced code block, or prose with a JSON
    object somewhere in it — people paste all three. Verdict synonyms are
    normalized.

    Raises:
        ValueError: when no JSON object with a recognizable verdict is found.
    """
    text = (payload or "").strip()
    if not text:
        raise ValueError("empty review payload")

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
            )
        idx = text.find("{", idx + 1)
    raise ValueError(f"could not extract a review decision from payload: {text[:200]!r}")


def _collect_context(repo: str, number: int, max_files: int = 50) -> dict[str, Any]:
    """Gather what a reviewer needs, using PyGithub."""
    from github import Auth, Github

    with Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])) as gh:
        pr = gh.get_repo(repo).get_pull(number)
        files = [
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                # Patches dominate the prompt; keep them for the first handful only.
                "patch": f.patch if i < 20 else None,
            }
            for i, f in enumerate(pr.get_files()[:max_files])
        ]
        return {
            "repo": repo,
            "number": number,
            "title": pr.title,
            "author": pr.user.login if pr.user else None,
            "body": pr.body or "",
            "base": pr.base.ref,
            "head": pr.head.ref,
            "url": pr.html_url,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
            "files": files,
            "prior_reviews": [{"user": r.user.login if r.user else None, "state": r.state} for r in pr.get_reviews()],
        }


def build_review_prompt(context: dict[str, Any], instructions: str = "") -> str:
    """Build the markdown prompt the reviewer sees in the Flyte UI.

    The metadata goes in a fenced JSON block so it renders verbatim and can be
    machine-read downstream.
    """
    instructions = instructions or (
        "Review this pull request. Respond with a JSON object of the form:\n"
        '`{"verdict": "approve" | "request_changes" | "comment", '
        '"summary": "...", "comments": [{"path": "...", "line": 1, '
        '"body": "...", "severity": "info" | "warning" | "blocking"}]}`'
    )
    return (
        f"## Review requested: {context['repo']}#{context['number']}\n\n"
        f"**{context['title']}** (by {context.get('author') or 'unknown'})\n\n"
        f"{context.get('body', '')}\n\n"
        f"{instructions}\n\n"
        "### Pull request metadata\n\n"
        "```json\n"
        f"{json.dumps(context, indent=2)}\n"
        "```\n"
    )


@env.task
async def review_pr(repo: str, number: int, timeout: timedelta | None = None) -> ReviewDecision:
    """Park the run on a human review condition and return the decision."""
    context = await asyncio.to_thread(_collect_context, repo, number)
    condition = await flyte.new_condition.aio(
        f"review-{repo.replace('/', '-')}-{number}"[:60],
        prompt=build_review_prompt(context),
        prompt_type="markdown",
        data_type=str,
        timeout=timeout,
    )
    return parse_review_payload(await condition.wait.aio())


@env.task
async def gated_merge(repo: str, number: int) -> str:
    """Wait for a human review, then merge if approved."""
    decision = await review_pr(repo, number)

    if not decision.is_approved:
        await asyncio.to_thread(_comment, repo, number, f"Review gate blocked this merge: {decision.summary}")
        return f"blocked: {decision.summary}"

    return await asyncio.to_thread(_merge, repo, number)


def _comment(repo: str, number: int, body: str) -> None:
    from github import Auth, Github

    with Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])) as gh:
        gh.get_repo(repo).get_issue(number).create_comment(body)


def _merge(repo: str, number: int) -> str:
    from github import Auth, Github

    with Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])) as gh:
        result = gh.get_repo(repo).get_pull(number).merge(merge_method="squash")
    return f"merged {result.sha}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext().run(gated_merge, repo="octocat/hello-world", number=1)
    print(run.url)
