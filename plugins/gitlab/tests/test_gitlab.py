"""GitLab-specific normalization, beyond what conformance covers."""

from __future__ import annotations

import json

from flyteplugins.gitlab import GitLabProvider, events, parse, payloads, verify

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


def test_pipeline_uses_status_as_the_action_and_keys_on_its_id():
    """A redelivery of the same pipeline status dedupes even without a delivery UUID."""

    def pipeline(uuid: str, status: str = "success"):
        return _parse(
            {
                "object_kind": "pipeline",
                "project": {"path_with_namespace": "octo/repo"},
                "object_attributes": {
                    "id": 31,
                    "iid": 4,
                    "status": status,
                    "created_at": "2024-01-01 00:00:00 UTC",
                    "finished_at": "2024-01-01 00:05:00 UTC",
                },
            },
            event="Pipeline Hook",
            uuid=uuid,
        )

    event = pipeline("u-1")
    assert event.qualified_type == events.Pipeline.SUCCESS
    assert event.resource_id == "octo/repo/-/pipelines/31"
    assert event.occurred_at == "2024-01-01 00:05:00 UTC"
    assert pipeline("u-1").dedupe_key() == pipeline("u-2").dedupe_key()
    assert pipeline("u-1").dedupe_key() != pipeline("u-1", status="failed").dedupe_key()


def test_deployment_reads_its_flat_payload():
    """GitLab sends deployment hooks with no `object_attributes`; `status` is top-level."""
    event = _parse(
        {
            "object_kind": "deployment",
            "status": "success",
            "status_changed_at": "2021-04-28 21:50:00 +0200",
            "deployment_id": 15,
            "deployable_id": 796,
            "deployable_url": "https://gitlab.com/octo/repo/-/jobs/796",
            "environment": "staging",
            "environment_tier": "staging",
            "project": {"id": 1, "path_with_namespace": "octo/repo"},
            "short_sha": "279484c0",
            "user": {"id": 1, "name": "Octocat", "username": "octocat"},
            "commit_title": "Add new file",
        },
        event="Deployment Hook",
    )
    assert event.qualified_type == events.Deployment.SUCCESS == "deployment.success"
    assert event.resource_id == "octo/repo/-/deployments/15"
    assert event.occurred_at == "2021-04-28 21:50:00 +0200"
    assert event.title == "staging"
    assert event.url == "https://gitlab.com/octo/repo/-/jobs/796"
    assert event.actor == "octocat"
    assert event.scope == "octo/repo"


def test_release_reads_its_flat_payload():
    """Release hooks carry `action` at the top level, not under `object_attributes`."""
    event = _parse(
        {
            "id": 1,
            "created_at": "2020-11-02 12:55:12 UTC",
            "description": "v1.1 has been released",
            "name": "v1.1 — the big one",
            "released_at": "2020-11-02 12:55:12 UTC",
            "tag": "v1.1",
            "object_kind": "release",
            "project": {"id": 1, "path_with_namespace": "octo/repo"},
            "url": "https://gitlab.com/octo/repo/-/releases/v1.1",
            "action": "create",
        },
        event="Release Hook",
    )
    assert event.qualified_type == events.Release.CREATE == "release.create"
    assert event.resource_id == "octo/repo/-/releases/v1.1"
    assert event.occurred_at == "2020-11-02 12:55:12 UTC"
    assert event.title == "v1.1 — the big one"
    assert event.url == "https://gitlab.com/octo/repo/-/releases/v1.1"


def test_every_constant_is_reachable_from_a_payload():
    """Each `kind.action` constant must be something `parse` can actually produce."""

    def spells(kind: str, action: str) -> str:
        if kind in ("deployment", "release"):
            field = "status" if kind == "deployment" else "action"
            payload = {"object_kind": kind, field: action, "project": {"path_with_namespace": "o/r"}}
        else:
            field = "status" if kind == "pipeline" else "action"
            payload = {"object_kind": kind, "object_attributes": {field: action}, "project": {}}
        return _parse(payload).qualified_type

    for cls in (events.MergeRequest, events.Issue, events.Pipeline, events.Release, events.Deployment):
        for member in cls:
            kind, _, action = member.value.partition(".")
            if action:
                assert spells(kind, action) == member, member


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


def test_push_payload_view_covers_the_commit_list():
    event = _parse(
        {
            "object_kind": "push",
            "ref": "refs/heads/main",
            "checkout_sha": "abc123",
            "user_username": "octocat",
            "project": {"path_with_namespace": "octo/repo"},
            "repository": {"name": "repo", "url": "git@gitlab.com:octo/repo.git"},
            "commits": [{"id": "abc123", "title": "Fix it", "author": {"name": "Octocat", "email": "o@x.io"}}],
            "total_commits_count": 1,
        },
        event="Push Hook",
    )
    view = payloads.push(event)
    assert view["commits"][0]["author"]["email"] == "o@x.io"
    assert view["repository"]["name"] == "repo"


def test_provider_is_not_signed_and_mounts_the_token_env():
    provider = GitLabProvider()
    assert provider.signed is False
    assert provider.default_secret_env == "GITLAB_WEBHOOK_TOKEN"
    assert provider.secret_env == "GITLAB_WEBHOOK_TOKEN"
