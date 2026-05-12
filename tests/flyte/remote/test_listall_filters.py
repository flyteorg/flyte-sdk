"""
Tests for the new project/domain/created_at/updated_at filter parameters
added to Run.listall() and Action.listall(), and for the TimeFilter/time_filtering
helpers in _common.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import list_pb2

from flyte.remote._common import TimeFilter, time_filtering

# ---------------------------------------------------------------------------
# TimeFilter / time_filtering unit tests
# ---------------------------------------------------------------------------


class TestTimeFilter:
    def test_defaults_to_none(self):
        tf = TimeFilter()
        assert tf.after is None
        assert tf.before is None

    def test_accepts_datetime_args(self):
        after = datetime(2024, 1, 1, tzinfo=timezone.utc)
        before = datetime(2024, 6, 1, tzinfo=timezone.utc)
        tf = TimeFilter(after=after, before=before)
        assert tf.after == after
        assert tf.before == before


class TestTimeFiltering:
    def test_empty_filter_produces_no_filters(self):
        result = time_filtering("created_at", TimeFilter())
        assert result == []

    def test_after_produces_gte_filter(self):
        after = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        result = time_filtering("created_at", TimeFilter(after=after))
        assert len(result) == 1
        f = result[0]
        assert f.function == list_pb2.Filter.Function.GREATER_THAN_OR_EQUAL
        assert f.field == "created_at"
        assert list(f.values) == [after.isoformat()]

    def test_before_produces_lt_filter(self):
        before = datetime(2024, 6, 1, tzinfo=timezone.utc)
        result = time_filtering("updated_at", TimeFilter(before=before))
        assert len(result) == 1
        f = result[0]
        assert f.function == list_pb2.Filter.Function.LESS_THAN
        assert f.field == "updated_at"
        assert list(f.values) == [before.isoformat()]

    def test_both_bounds_produce_two_filters(self):
        after = datetime(2024, 1, 1, tzinfo=timezone.utc)
        before = datetime(2024, 12, 31, tzinfo=timezone.utc)
        result = time_filtering("created_at", TimeFilter(after=after, before=before))
        assert len(result) == 2
        functions = {f.function for f in result}
        assert list_pb2.Filter.Function.GREATER_THAN_OR_EQUAL in functions
        assert list_pb2.Filter.Function.LESS_THAN in functions

    def test_naive_datetime_is_converted_to_utc(self):
        # A timezone-aware datetime should come out as UTC ISO string.
        dt = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = time_filtering("created_at", TimeFilter(after=dt))
        assert result[0].values[0] == "2024-03-15T12:00:00+00:00"

    def test_field_name_is_forwarded(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for field in ("created_at", "updated_at", "start_time"):
            result = time_filtering(field, TimeFilter(after=dt))
            assert result[0].field == field


# ---------------------------------------------------------------------------
# Run.listall() — verify filters sent to the gRPC stub
# ---------------------------------------------------------------------------


def _make_run_pb2():
    """Build a minimal run_definition_pb2.Run so Run() can be instantiated."""
    from flyteidl2.common import identifier_pb2, phase_pb2
    from flyteidl2.workflow import run_definition_pb2

    action = run_definition_pb2.Action(
        id=identifier_pb2.ActionIdentifier(
            run=identifier_pb2.RunIdentifier(
                org="test-org",
                project="test-project",
                domain="development",
                name="run-abc",
            ),
            name="a0",
        ),
        status=run_definition_pb2.ActionStatus(
            phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        ),
    )
    return run_definition_pb2.Run(action=action)


def _make_list_runs_response(runs=None):
    from flyteidl2.workflow import run_service_pb2

    resp = MagicMock(spec=run_service_pb2.ListRunsResponse)
    resp.runs = runs or []
    resp.token = ""
    return resp


@pytest.fixture
def mock_init_config():
    cfg = MagicMock()
    cfg.org = "test-org"
    cfg.project = "default-project"
    cfg.domain = "production"
    return cfg


@pytest.fixture
def mock_run_service():
    svc = MagicMock()
    svc.list_runs = AsyncMock(return_value=_make_list_runs_response())
    return svc


@pytest.fixture
def mock_client(mock_run_service):
    client = MagicMock()
    client.run_service = mock_run_service
    return client


class TestRunListallFilters:
    """Verify that Run.listall() forwards project/domain/time filters to the gRPC stub."""

    def _call_listall(self, mock_client, mock_init_config, **kwargs):
        with (
            patch("flyte.remote._run.ensure_client"),
            patch("flyte.remote._run.get_client", return_value=mock_client),
            patch("flyte.remote._run.get_init_config", return_value=mock_init_config),
        ):
            from flyte.remote._run import Run

            list(Run.listall(**kwargs))

        return mock_client.run_service.list_runs.call_args

    def test_defaults_use_config_project_and_domain(self, mock_client, mock_init_config):
        call_args = self._call_listall(mock_client, mock_init_config)
        req = call_args[0][0]
        assert req.project_id.name == "default-project"
        assert req.project_id.domain == "production"

    def test_explicit_project_overrides_config(self, mock_client, mock_init_config):
        call_args = self._call_listall(mock_client, mock_init_config, project="customer-us-east")
        req = call_args[0][0]
        assert req.project_id.name == "customer-us-east"

    def test_explicit_domain_overrides_config(self, mock_client, mock_init_config):
        call_args = self._call_listall(mock_client, mock_init_config, domain="staging")
        req = call_args[0][0]
        assert req.project_id.domain == "staging"

    def test_both_project_and_domain_override(self, mock_client, mock_init_config):
        call_args = self._call_listall(mock_client, mock_init_config, project="proj-x", domain="dev")
        req = call_args[0][0]
        assert req.project_id.name == "proj-x"
        assert req.project_id.domain == "dev"

    def test_created_at_filter_is_sent(self, mock_client, mock_init_config):
        after = datetime(2024, 1, 1, tzinfo=timezone.utc)
        call_args = self._call_listall(mock_client, mock_init_config, created_at=TimeFilter(after=after))
        filters = list(call_args[0][0].request.filters)
        fields = [f.field for f in filters]
        assert "created_at" in fields
        gte = next(f for f in filters if f.field == "created_at")
        assert gte.function == list_pb2.Filter.Function.GREATER_THAN_OR_EQUAL
        assert list(gte.values) == [after.isoformat()]

    def test_updated_at_filter_is_sent(self, mock_client, mock_init_config):
        before = datetime(2024, 6, 1, tzinfo=timezone.utc)
        call_args = self._call_listall(mock_client, mock_init_config, updated_at=TimeFilter(before=before))
        filters = list(call_args[0][0].request.filters)
        fields = [f.field for f in filters]
        assert "updated_at" in fields
        lt = next(f for f in filters if f.field == "updated_at")
        assert lt.function == list_pb2.Filter.Function.LESS_THAN

    def test_time_filters_combine_with_phase_filter(self, mock_client, mock_init_config):
        after = datetime(2024, 1, 1, tzinfo=timezone.utc)
        call_args = self._call_listall(
            mock_client,
            mock_init_config,
            in_phase=("succeeded",),
            created_at=TimeFilter(after=after),
        )
        filters = list(call_args[0][0].request.filters)
        fields = [f.field for f in filters]
        assert "phase" in fields
        assert "created_at" in fields

    def test_no_time_filters_sends_no_time_fields(self, mock_client, mock_init_config):
        call_args = self._call_listall(mock_client, mock_init_config)
        filters = list(call_args[0][0].request.filters)
        fields = [f.field for f in filters]
        assert "created_at" not in fields
        assert "updated_at" not in fields


# ---------------------------------------------------------------------------
# Action.listall() — verify time filters sent to the gRPC stub
# ---------------------------------------------------------------------------


def _make_list_actions_response():
    resp = MagicMock()
    resp.actions = []
    resp.token = ""
    return resp


@pytest.fixture
def mock_action_service():
    svc = MagicMock()
    svc.list_actions = AsyncMock(return_value=_make_list_actions_response())
    return svc


@pytest.fixture
def mock_client_action(mock_action_service):
    client = MagicMock()
    client.run_service = mock_action_service
    return client


class TestActionListallFilters:
    """Verify that Action.listall() forwards time filters to the gRPC stub."""

    def _call_listall(self, mock_client, mock_init_config, **kwargs):
        with (
            patch("flyte.remote._action.ensure_client"),
            patch("flyte.remote._action.get_client", return_value=mock_client),
            patch("flyte.remote._action.get_init_config", return_value=mock_init_config),
        ):
            from flyte.remote._action import Action

            list(Action.listall(for_run_name="run-abc", **kwargs))

        return mock_client.run_service.list_actions.call_args

    def test_created_at_filter_is_sent(self, mock_client_action, mock_init_config):
        after = datetime(2024, 3, 1, tzinfo=timezone.utc)
        call_args = self._call_listall(mock_client_action, mock_init_config, created_at=TimeFilter(after=after))
        filters = list(call_args[0][0].request.filters)
        fields = [f.field for f in filters]
        assert "created_at" in fields
        gte = next(f for f in filters if f.field == "created_at")
        assert gte.function == list_pb2.Filter.Function.GREATER_THAN_OR_EQUAL
        assert list(gte.values) == [after.isoformat()]

    def test_updated_at_filter_is_sent(self, mock_client_action, mock_init_config):
        before = datetime(2024, 9, 1, tzinfo=timezone.utc)
        call_args = self._call_listall(mock_client_action, mock_init_config, updated_at=TimeFilter(before=before))
        filters = list(call_args[0][0].request.filters)
        fields = [f.field for f in filters]
        assert "updated_at" in fields

    def test_both_time_filters_are_sent(self, mock_client_action, mock_init_config):
        after = datetime(2024, 1, 1, tzinfo=timezone.utc)
        before = datetime(2024, 12, 31, tzinfo=timezone.utc)
        call_args = self._call_listall(
            mock_client_action,
            mock_init_config,
            created_at=TimeFilter(after=after, before=before),
        )
        filters = list(call_args[0][0].request.filters)
        created_at_filters = [f for f in filters if f.field == "created_at"]
        assert len(created_at_filters) == 2

    def test_time_filters_combine_with_phase_filter(self, mock_client_action, mock_init_config):
        after = datetime(2024, 1, 1, tzinfo=timezone.utc)
        call_args = self._call_listall(
            mock_client_action,
            mock_init_config,
            in_phase=("running",),
            updated_at=TimeFilter(after=after),
        )
        filters = list(call_args[0][0].request.filters)
        fields = [f.field for f in filters]
        assert "phase" in fields
        assert "updated_at" in fields

    def test_no_time_filters_sends_no_time_fields(self, mock_client_action, mock_init_config):
        call_args = self._call_listall(mock_client_action, mock_init_config)
        # filters may be None when filter_list is empty
        raw_filters = call_args[0][0].request.filters
        if raw_filters:
            fields = [f.field for f in raw_filters]
            assert "created_at" not in fields
            assert "updated_at" not in fields


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


class TestTimeFilterExport:
    def test_importable_from_flyte_remote(self):
        from flyte.remote import TimeFilter as TF

        assert TF is TimeFilter
