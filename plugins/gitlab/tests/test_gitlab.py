"""GitLab-specific normalization, beyond what conformance covers."""

from __future__ import annotations

import json

from flyteplugins.gitlab import GitLabProvider, events, parse, verify

TOKEN = "gl-secret"


def _headers(body: bytes, event: str = "Merge Request Hook", uuid: str = "u-1") -> dict:
    return {
        "X-Gitlab-Event": event,
        "X-Gitlab-Event-UUID": uuid,
        "X-Gitlab-Token": TOKEN,
    }


def _parse(payload: dict, event: str = "Merge Request Hook", uuid: str = "u-1") -> object:
    body = json.dumps(payload).encode()
    return parse(_headers(body, event=event, uuid=uuid), body)


def test_verify_compares_the_shared_token():
    body = b"{}"
    assert verify(body, {"X-Gitlab-Token": TOKEN}, TOKEN) is True
    assert verify(body, {"X-Gitlab-Token": "nope"}, TOKEN) is False
    assert verify(body, {}, TOKEN) is False


def test_merge_request_normalizes_to_the_constant():
    event = _parse(
        {
            "object_kind": "merge_request",
            "user": {"username": "octocat"},
            "project": {"path_with_namespace": "octo/repo"},
            "object_attributes": {
                "iid": 7,
                "action": "open",
                "title": "Add a feature",
                "source_branch": "feature",
            },
        }
    )
    assert event.qualified_type == events.MergeRequest.OPEN
    assert event.resource_id == "octo/repo!7"
    assert event.scope == "octo/repo"
    assert event.actor == "octocat"


def test_issue_uses_the_hash_marker():
    event = _parse(
        {
            "object_kind": "issue",
            "project": {"path_with_namespace": "octo/repo"},
            "object_attributes": {"iid": 3, "action": "open"},
        },
        event="Issue Hook",
    )
    assert event.qualified_type == events.Issue.OPEN
    assert event.resource_id == "octo/repo#3"


def test_distinct_notes_on_one_thread_do_not_collapse():
    """Keyed on the thread alone, every comment after the first looks like a redelivery."""

    def note(note_id: int):
        return _parse(
            {
                "object_kind": "note",
                "project": {"path_with_namespace": "octo/repo"},
                "object_attributes": {"id": note_id, "noteable_type": "MergeRequest"},
                "merge_request": {"iid": 7, "title": "Add a feature"},
            },
            event="Note Hook",
        )

    assert note(11).dedupe_key() != note(12).dedupe_key()
    assert note(11).dedupe_key() == note(11).dedupe_key()
    assert note(11).qualified_type == events.Note.ANY == "note"


def test_a_note_surfaces_its_parents_title():
    """Comments carry no title of their own; the dashboard should show the thread's."""
    event = _parse(
        {
            "object_kind": "note",
            "project": {"path_with_namespace": "octo/repo"},
            "object_attributes": {"id": 42, "noteable_type": "MergeRequest"},
            "merge_request": {"iid": 7, "title": "Add a feature"},
        },
        event="Note Hook",
    )
    assert event.title == "Add a feature"


def test_merge_request_actions_track_gitlabs_wire_values():
    """`draft` is a boolean field, not an action; approval comes as approval/approved."""
    assert events.MergeRequest.APPROVAL == "merge_request.approval"
    assert events.MergeRequest.APPROVED == "merge_request.approved"
    assert events.MergeRequest.UNAPPROVAL == "merge_request.unapproval"
    assert events.MergeRequest.UNAPPROVED == "merge_request.unapproved"


def test_a_note_on_an_issue_keys_with_the_hash_marker():
    event = _parse(
        {
            "object_kind": "note",
            "project": {"path_with_namespace": "octo/repo"},
            "object_attributes": {"id": 42, "noteable_type": "Issue"},
            "issue": {"iid": 3, "title": "A bug"},
        },
        event="Note Hook",
    )
    assert event.resource_id == "octo/repo#3:42"


def test_pipeline_uses_status_as_the_action():
    event = _parse(
        {
            "object_kind": "pipeline",
            "project": {"path_with_namespace": "octo/repo"},
            "object_attributes": {"id": 31, "status": "success"},
        },
        event="Pipeline Hook",
    )
    assert event.qualified_type == events.Pipeline.SUCCESS


def test_push_has_no_resource_and_falls_back_to_the_delivery_id():
    def push(uuid: str):
        return _parse(
            {
                "object_kind": "push",
                "project": {"path_with_namespace": "octo/repo"},
                "user_username": "octocat",
            },
            event="Push Hook",
            uuid=uuid,
        )

    a, b = push("x"), push("y")
    assert a.qualified_type == events.Push.ANY == "push"
    assert a.resource_id is None
    assert a.actor == "octocat"
    assert a.dedupe_key() != b.dedupe_key()


def test_provider_is_not_signed_and_mounts_the_token_env():
    provider = GitLabProvider()
    assert provider.signed is False
    assert provider.default_secret_env == "GITLAB_WEBHOOK_TOKEN"
    assert provider.secret_env == "GITLAB_WEBHOOK_TOKEN"
