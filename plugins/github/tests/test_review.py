"""Tests for the PR review condition helpers."""

from __future__ import annotations

import json

import pytest

from flyteplugins.github import (
    ReviewContext,
    build_review_prompt,
    parse_review_payload,
)


def make_context() -> ReviewContext:
    return ReviewContext(
        repo="octo/repo",
        number=42,
        title="Add feature",
        author="octocat",
        body="Implements the feature.",
        base="main",
        head="feature",
        url="https://github.com/octo/repo/pull/42",
        additions=10,
        deletions=2,
        changed_files=3,
        files=[{"filename": "src/app.py", "additions": 10, "deletions": 2, "patch": "@@ ... +10 -2"}],
    )


def test_build_review_prompt_embeds_json():
    prompt = build_review_prompt(make_context())
    assert "review requested" in prompt.lower()
    assert "```json" in prompt
    # the embedded JSON must round-trip
    block = prompt.split("```json")[1].split("```")[0]
    data = json.loads(block)
    assert data["number"] == 42
    assert data["files"][0]["filename"] == "src/app.py"


def test_build_review_prompt_truncates_patches():
    context = make_context()
    context.files = [{"filename": f"f{i}.py", "patch": "x" * 100} for i in range(25)]
    block = build_review_prompt(context).split("```json")[1].split("```")[0]
    data = json.loads(block)
    assert "patch" in data["files"][0]
    assert "patch" not in data["files"][20]


def test_parse_review_payload_plain_json():
    decision = parse_review_payload('{"verdict": "approve", "summary": "looks good", "reviewer": "alice"}')
    assert decision.is_approved
    assert decision.summary == "looks good"
    assert decision.reviewer == "alice"


def test_parse_review_payload_json_in_fence():
    payload = (
        "Here is my review:\n\n```json\n"
        '{"verdict": "request_changes", "summary": "needs tests", '
        '"comments": [{"path": "a.py", "line": 4, "body": "untested", "severity": "blocking"}]}'
        "\n```\n"
    )
    decision = parse_review_payload(payload)
    assert decision.verdict == "request_changes"
    assert len(decision.blocking_comments) == 1
    assert decision.blocking_comments[0].path == "a.py"


def test_parse_review_payload_json_in_prose():
    decision = parse_review_payload('After careful thought I say {"verdict": "approved", "summary": "ship it"} thanks!')
    assert decision.verdict == "approve"


def test_parse_review_payload_verdict_synonyms():
    for raw, expected in [
        ("lgtm", "approve"),
        ("APPROVED", "approve"),
        ("changes_requested", "request_changes"),
        ("neutral", "comment"),
    ]:
        decision = parse_review_payload(json.dumps({"verdict": raw}))
        assert decision.verdict == expected


def test_parse_review_payload_rejects_garbage():
    with pytest.raises(ValueError):
        parse_review_payload("no json here at all")
    with pytest.raises(ValueError):
        parse_review_payload("")
    with pytest.raises(ValueError):
        parse_review_payload('{"verdict": "maybe"}')
