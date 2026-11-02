"""Telling a client-side failure apart from a service outage when a call comes back UNAVAILABLE."""

import platform

import click
import pytest
from click.testing import CliRunner
from connectrpc.code import Code
from connectrpc.codec import proto_binary_codec
from connectrpc.errors import ConnectError
from flyteidl2.project import project_service_connect, project_service_pb2

from flyte.cli._common import CommandBase
from flyte.remote._client._connect_errors import describe_local_failure, local_failure


class _BrokenCodec:
    """A codec that fails to encode, standing in for any library failing inside this process."""

    def __init__(self):
        self._inner = proto_binary_codec()

    def name(self):
        return self._inner.name()

    def encode(self, message):
        raise TypeError("cannot encode this message")

    def decode(self, data, message):
        return self._inner.decode(data, message)


async def _list_projects(client) -> ConnectError:
    with pytest.raises(ConnectError) as exc:
        await client.list_projects(project_service_pb2.ListProjectsRequest())
    return exc.value


@pytest.mark.asyncio
async def test_local_exception_inside_connectrpc_is_a_local_failure():
    # connectrpc turns the codec's TypeError into UNAVAILABLE, which reads like an outage.
    client = project_service_connect.ProjectServiceClient("http://127.0.0.1:1", codec=_BrokenCodec())
    err = await _list_projects(client)

    assert err.code == Code.UNAVAILABLE
    cause = local_failure(err)
    assert isinstance(cause, TypeError)


@pytest.mark.asyncio
async def test_refused_connection_is_not_a_local_failure():
    client = project_service_connect.ProjectServiceClient("http://127.0.0.1:1")
    err = await _list_projects(client)

    assert err.code == Code.UNAVAILABLE
    assert local_failure(err) is None


def test_unavailable_from_the_server_is_not_a_local_failure():
    assert local_failure(ConnectError(Code.UNAVAILABLE, "no healthy cluster")) is None


def test_other_codes_are_not_local_failures():
    try:
        raise ConnectError(Code.INTERNAL, "boom") from TypeError("x")
    except ConnectError as e:
        assert local_failure(e) is None


def test_describe_local_failure_names_the_exception_and_python_version():
    msg = describe_local_failure(TypeError("cannot encode"), "Submitting the run")

    assert "TypeError: cannot encode" in msg
    assert f"Python {platform.python_version()}" in msg
    assert "unavailable" not in msg.lower().replace("not a service outage", "")


def _cli_raising(exc: BaseException) -> click.Command:
    @click.command(cls=CommandBase)
    def cmd(**_kwargs):
        raise exc

    return cmd


def test_cli_reports_local_failure_instead_of_unavailable():
    try:
        raise ConnectError(Code.UNAVAILABLE, "cannot encode") from TypeError("cannot encode")
    except ConnectError as e:
        err = e

    result = CliRunner().invoke(_cli_raising(err))

    assert result.exit_code != 0
    assert "TypeError: cannot encode" in result.output
    assert "currently unavailable" not in result.output


def test_cli_still_reports_server_unavailable():
    result = CliRunner().invoke(_cli_raising(ConnectError(Code.UNAVAILABLE, "tunnel down")))

    assert "Service is currently unavailable" in result.output
    assert "tunnel down" in result.output
