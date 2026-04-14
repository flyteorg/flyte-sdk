"""Tests for Project.create, Project.update, Project.archive/unarchive, and Project.listall state filtering."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.project import project_service_pb2
from flyteidl2.task import run_pb2

from flyte.remote._project import Project


def _make_project_pb(
    id: str = "test-project",
    name: str = "test-project",
    description: str = "A test project",
    state: int = project_service_pb2.PROJECT_STATE_ACTIVE,
    labels: dict | None = None,
) -> project_service_pb2.Project:
    pb = project_service_pb2.Project(id=id, name=name, description=description, state=state)
    if labels:
        pb.labels.CopyFrom(run_pb2.Labels(values=labels))
    return pb


def _make_get_response(project_pb):
    resp = MagicMock()
    resp.project = project_pb
    return resp


def _make_list_response(projects, token=""):
    resp = MagicMock()
    resp.projects.projects = projects
    resp.projects.token = token
    return resp


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.create_project = AsyncMock(return_value=project_service_pb2.CreateProjectResponse())
    svc.update_project = AsyncMock(return_value=project_service_pb2.UpdateProjectResponse())
    svc.get_project = AsyncMock()
    svc.list_projects = AsyncMock()
    return svc


@pytest.fixture
def mock_client(mock_service):
    client = MagicMock()
    client.project_domain_service = mock_service
    return client


class TestProjectCreate:
    def test_create_minimal(self, mock_client, mock_service):
        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            result = Project.create(id="my-project-id", name="My Project")

        mock_service.create_project.assert_called_once()
        req = mock_service.create_project.call_args[0][0]
        assert req.project.id == "my-project-id"
        assert req.project.name == "My Project"
        assert req.project.description == ""
        assert result.pb2.id == "my-project-id"
        assert result.pb2.name == "My Project"

    def test_create_with_description(self, mock_client, mock_service):
        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.create(id="my-project-id", name="My Project", description="My cool project")

        req = mock_service.create_project.call_args[0][0]
        assert req.project.description == "My cool project"

    def test_create_with_labels(self, mock_client, mock_service):
        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.create(id="my-project-id", name="My Project", labels={"team": "ml", "env": "prod"})

        req = mock_service.create_project.call_args[0][0]
        assert dict(req.project.labels.values) == {"team": "ml", "env": "prod"}

    def test_create_without_labels_does_not_set_labels(self, mock_client, mock_service):
        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.create(id="my-project-id", name="My Project")

        req = mock_service.create_project.call_args[0][0]
        assert not req.project.labels.values


class TestProjectUpdate:
    def test_update_description(self, mock_client, mock_service):
        existing = _make_project_pb(description="old desc")
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.update(id="test-project", description="new desc")

        mock_service.update_project.assert_called_once()
        req = mock_service.update_project.call_args[0][0]
        assert req.project.description == "new desc"
        # Name should be preserved
        assert req.project.name == "test-project"

    def test_update_name(self, mock_client, mock_service):
        existing = _make_project_pb(name="Old Name")
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.update(id="test-project", name="New Name")

        req = mock_service.update_project.call_args[0][0]
        assert req.project.name == "New Name"

    def test_update_labels(self, mock_client, mock_service):
        existing = _make_project_pb(labels={"old": "label"})
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.update(id="test-project", labels={"new": "label"})

        req = mock_service.update_project.call_args[0][0]
        assert dict(req.project.labels.values) == {"new": "label"}

    def test_update_state_archived(self, mock_client, mock_service):
        existing = _make_project_pb()
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.update(id="test-project", state="archived")

        req = mock_service.update_project.call_args[0][0]
        assert req.project.state == project_service_pb2.PROJECT_STATE_ARCHIVED

    def test_update_state_active(self, mock_client, mock_service):
        existing = _make_project_pb(state=project_service_pb2.PROJECT_STATE_ARCHIVED)
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.update(id="test-project", state="active")

        req = mock_service.update_project.call_args[0][0]
        assert req.project.state == project_service_pb2.PROJECT_STATE_ACTIVE

    def test_update_preserves_unmodified_fields(self, mock_client, mock_service):
        existing = _make_project_pb(
            name="Original Name",
            description="Original desc",
            labels={"keep": "me"},
        )
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.update(id="test-project", state="archived")

        req = mock_service.update_project.call_args[0][0]
        # Only state should change; other fields preserved
        assert req.project.name == "Original Name"
        assert req.project.description == "Original desc"
        assert dict(req.project.labels.values) == {"keep": "me"}
        assert req.project.state == project_service_pb2.PROJECT_STATE_ARCHIVED

    def test_update_fetches_project_by_id(self, mock_client, mock_service):
        existing = _make_project_pb()
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.update(id="my-proj", description="x")

        get_req = mock_service.get_project.call_args[0][0]
        assert get_req.id == "my-proj"

    def test_update_clears_domains(self, mock_client, mock_service):
        existing = _make_project_pb()
        existing.domains.append(project_service_pb2.Domain(id="production", name="production"))
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            Project.update(id="test-project", description="x")

        req = mock_service.update_project.call_args[0][0]
        assert len(req.project.domains) == 0


class TestProjectArchiveUnarchive:
    def test_archive(self, mock_client, mock_service):
        existing = _make_project_pb()
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            project = Project(existing)
            project.archive()

        req = mock_service.update_project.call_args[0][0]
        assert req.project.state == project_service_pb2.PROJECT_STATE_ARCHIVED

    def test_unarchive(self, mock_client, mock_service):
        existing = _make_project_pb(state=project_service_pb2.PROJECT_STATE_ARCHIVED)
        mock_service.get_project.return_value = _make_get_response(existing)

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            project = Project(existing)
            project.unarchive()

        req = mock_service.update_project.call_args[0][0]
        assert req.project.state == project_service_pb2.PROJECT_STATE_ACTIVE


class TestProjectListallStateFilter:
    def _call_listall(self, mock_client, archived=False):
        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            return list(Project.listall(archived=archived))

    def test_default_sends_active_filter(self, mock_client, mock_service):
        mock_service.list_projects.return_value = _make_list_response([_make_project_pb()])

        self._call_listall(mock_client)

        req = mock_service.list_projects.call_args[0][0]
        assert req.filters == f"eq(state, {project_service_pb2.PROJECT_STATE_ACTIVE})"

    def test_archived_sends_archived_filter(self, mock_client, mock_service):
        mock_service.list_projects.return_value = _make_list_response([])

        self._call_listall(mock_client, archived=True)

        req = mock_service.list_projects.call_args[0][0]
        assert req.filters == f"eq(state, {project_service_pb2.PROJECT_STATE_ARCHIVED})"

    def test_combines_with_existing_filters(self, mock_client, mock_service):
        mock_service.list_projects.return_value = _make_list_response([])

        with (
            patch("flyte.remote._project.ensure_client"),
            patch("flyte.remote._project.get_client", return_value=mock_client),
        ):
            list(Project.listall(filters="eq(name, foo)", archived=True))

        req = mock_service.list_projects.call_args[0][0]
        assert req.filters == f"eq(name, foo)+eq(state, {project_service_pb2.PROJECT_STATE_ARCHIVED})"

    def test_listall_paginates(self, mock_client, mock_service):
        p1 = _make_project_pb(id="proj-1")
        p2 = _make_project_pb(id="proj-2")
        mock_service.list_projects.side_effect = [
            _make_list_response([p1], token="next"),
            _make_list_response([p2], token=""),
        ]

        results = self._call_listall(mock_client)
        assert len(results) == 2
        assert mock_service.list_projects.call_count == 2


class TestProjectRichRepr:
    def test_rich_repr_active(self):
        pb = _make_project_pb(labels={"team": "ml"})
        project = Project(pb)
        fields = dict(project.__rich_repr__())
        assert fields["name"] == "test-project"
        assert fields["id"] == "test-project"
        assert fields["description"] == "A test project"
        assert fields["state"] == "PROJECT_STATE_ACTIVE"
        assert "team: ml" in fields["labels"]

    def test_rich_repr_no_labels(self):
        pb = _make_project_pb()
        project = Project(pb)
        fields = dict(project.__rich_repr__())
        assert not fields["labels"]
