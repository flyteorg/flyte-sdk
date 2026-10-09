"""`flyte.remote.Task.get(...)` scope properties, and `TaskService.list_versions` on the client protocol."""

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from flyte.remote import Task
from flyte.remote._task import LazyEntity


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"project": "p", "domain": "d", "version": "v1"}, ("p", "d", "v1", None)),
        ({"auto_version": "latest"}, (None, None, None, "latest")),
        ({"project": "p", "auto_version": "current"}, ("p", None, None, "current")),
    ],
)
def test_scope_properties_without_fetch(kwargs, expected):
    ref = Task.get("env.t", **kwargs)
    assert isinstance(ref, LazyEntity)
    with patch.object(ref, "_getter", AsyncMock(side_effect=AssertionError("must not fetch"))):
        assert (ref.project, ref.domain, ref.version, ref.auto_version) == expected
        assert ref.name == "env.t"
    assert ref._task is None


def test_scope_survives_override():
    ref = Task.get("env.t", project="p", domain="d", version="v1")
    fake_details = AsyncMock()
    fake_details.override = lambda **kw: "overridden"
    with patch.object(ref, "_getter", AsyncMock(return_value=fake_details)):
        new = ref.override(short_name="x")
    assert (new.project, new.domain, new.version) == ("p", "d", "v1")


def test_lazy_entity_positional_construction_still_works():
    async def getter():
        return None

    e = LazyEntity("n", getter)
    assert (e.name, e.project, e.domain, e.version, e.auto_version) == ("n", None, None, None, None)


def test_task_service_protocol_has_list_versions():
    from flyteidl2.task import task_service_pb2
    from flyteidl2.task.task_service_connect import TaskServiceClient

    from flyte.remote._client._protocols import TaskService

    sig = inspect.signature(TaskService.list_versions)
    assert sig.parameters["request"].annotation in (
        task_service_pb2.ListVersionsRequest,
        "task_service_pb2.ListVersionsRequest",
    )
    # The controlplane client is the generated connect client, which implements the RPC.
    assert callable(getattr(TaskServiceClient, "list_versions", None))
