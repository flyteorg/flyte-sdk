"""Tests for the pull-request review gate."""

from __future__ import annotations

import pytest

from flyteplugins.github import (
    ReviewContext,
    ReviewDecision,
    build_review_prompt,
    condition_name_for,
    parse_review_payload,
)


def _context(**kwargs) -> ReviewContext:
    base = {"repo": "octo/repo", "number": 7, "title": "Add a feature", "author": "octocat"}
    return ReviewContext(**{**base, **kwargs})


# ----------------------------------------------------------------------
# parsing a reviewer's answer
# ----------------------------------------------------------------------


def test_raw_json_is_parsed():
    decision = parse_review_payload('{"verdict": "approve", "summary": "ship it"}')
    assert decision.is_approved
    assert decision.summary == "ship it"


def test_json_in_a_fenced_block_is_parsed():
    """Reviewers paste from the UI, which often wraps the answer in a fence."""
    payload = 'Looks fine.\n\n```json\n{"verdict": "approve", "summary": "ok"}\n```\n'
    assert parse_review_payload(payload).is_approved


def test_json_buried_in_prose_is_parsed():
    payload = 'I looked at this and {"verdict": "request_changes", "summary": "needs tests"} is my call.'
    decision = parse_review_payload(payload)
    assert decision.verdict == "request_changes"
    assert not decision.is_approved


def test_the_first_object_without_a_verdict_is_skipped():
    """A prompt echo can contain JSON of its own before the real answer."""
    payload = '{"note": "not the answer"} then {"verdict": "approve"}'
    assert parse_review_payload(payload).is_approved


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("approved", "approve"),
        ("LGTM", "approve"),
        ("accept", "approve"),
        ("changes_requested", "request_changes"),
        ("Changes Requested", "request_changes"),
        ("reject", "request_changes"),
        ("blocked", "request_changes"),
        ("comment", "comment"),
        ("neutral", "comment"),
    ],
)
def test_verdict_synonyms_are_normalized(written, expected):
    assert parse_review_payload(f'{{"verdict": "{written}"}}').verdict == expected


def test_inline_comments_are_typed():
    payload = """
    {"verdict": "request_changes", "comments": [
        {"path": "a.py", "line": 3, "body": "leak", "severity": "blocking"},
        {"path": "b.py", "body": "nit"}
    ]}
    """
    decision = parse_review_payload(payload)
    assert [c.path for c in decision.comments] == ["a.py", "b.py"]
    assert [c.severity for c in decision.blocking_comments] == ["blocking"]
    assert decision.comments[1].severity == "info", "severity should default to info"


def test_an_empty_or_verdictless_payload_raises():
    for payload in ("", "   ", "no json here", '{"summary": "forgot the verdict"}'):
        with pytest.raises(ValueError):
            parse_review_payload(payload)


def test_an_unknown_verdict_raises():
    with pytest.raises(ValueError, match="unknown verdict"):
        parse_review_payload('{"verdict": "maybe?"}')


# ----------------------------------------------------------------------
# the prompt a reviewer sees
# ----------------------------------------------------------------------


def test_the_prompt_carries_the_metadata_as_parseable_json():
    """The reviewer reads the markdown; downstream code reads the JSON block."""
    import json

    context = _context(files=[{"filename": "a.py", "patch": "@@"}])
    prompt = build_review_prompt(context)
    assert "octo/repo#7" in prompt
    body = prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(body)["files"][0]["filename"] == "a.py"


def test_patches_are_dropped_past_the_cap_but_stats_are_kept():
    """A large diff would otherwise dominate the prompt."""
    import json

    context = _context(files=[{"filename": f"f{i}.py", "patch": "x" * 50, "additions": 1} for i in range(30)])
    data = json.loads(context.to_json(max_file_patches=5))
    assert data["files"][4]["patch"] is not None
    assert "patch" not in data["files"][5]
    assert data["files"][5]["additions"] == 1, "stats survive; only the patch is dropped"


def test_custom_instructions_replace_the_default():
    prompt = build_review_prompt(_context(), instructions="Only check the migration.")
    assert "Only check the migration." in prompt
    assert "Respond with a JSON object" not in prompt


# ----------------------------------------------------------------------
# condition naming
# ----------------------------------------------------------------------


def test_a_condition_name_is_derived_from_the_pull_request():
    assert condition_name_for("octo/repo", 7) == "review-octo-repo-7"


def test_a_long_repo_name_is_truncated_but_keeps_the_number():
    """Condition names are action names on the backend, so they are bounded."""
    name = condition_name_for("a-very-long-organisation/an-even-longer-repository-name", 4242)
    assert len(name) <= 60
    assert name.endswith("-4242"), "the number is what distinguishes one review from the next"


def test_two_pull_requests_never_share_a_condition_name():
    long_repo = "a-very-long-organisation/an-even-longer-repository-name"
    assert condition_name_for(long_repo, 1) != condition_name_for(long_repo, 2)


# ----------------------------------------------------------------------
# the decision the workflow branches on
# ----------------------------------------------------------------------


def test_is_approved_only_for_approve():
    assert ReviewDecision(verdict="approve").is_approved
    assert not ReviewDecision(verdict="request_changes").is_approved
    assert not ReviewDecision(verdict="comment").is_approved
