import pathlib

import pytest

from flyte._image import Image
from flyte.app import AppEnvironment, ConnectorEnvironment, Port
from flyte.models import SerializationContext


def test_connector_defaults():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"))
    assert conn.type == "connector"
    assert isinstance(conn.port, Port)
    assert conn.port.port == 8080
    assert conn.port.name == "h2c"


def test_connector_is_app_environment():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"))
    assert isinstance(conn, AppEnvironment)


def test_connector_custom_port_int():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"), port=9000)
    assert isinstance(conn.port, Port)
    assert conn.port.port == 9000


def test_connector_custom_port_object():
    port = Port(port=8888, name="custom-h2c")
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"), port=port)
    assert conn.port == port


def test_connector_default_args():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"))
    ctx = SerializationContext(org="org", project="proj", domain="dev", version="v1", root_dir=pathlib.Path.cwd())
    args = conn.container_args(ctx)
    assert args == ["c0", "--port", "8080", "--prometheus_port", "9092"]


def test_connector_default_args_custom_port():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"), port=9999)
    ctx = SerializationContext(org="org", project="proj", domain="dev", version="v1", root_dir=pathlib.Path.cwd())
    args = conn.container_args(ctx)
    assert args == ["c0", "--port", "9999", "--prometheus_port", "9092"]


def test_connector_custom_args_list():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"), args=["--custom", "arg"])
    ctx = SerializationContext(org="org", project="proj", domain="dev", version="v1", root_dir=pathlib.Path.cwd())
    args = conn.container_args(ctx)
    assert args == ["--custom", "arg"]


def test_connector_custom_args_string():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"), args="--host 0.0.0.0")
    ctx = SerializationContext(org="org", project="proj", domain="dev", version="v1", root_dir=pathlib.Path.cwd())
    args = conn.container_args(ctx)
    assert args == ["--host", "0.0.0.0"]


def test_connector_empty_args():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"), args=[])
    ctx = SerializationContext(org="org", project="proj", domain="dev", version="v1", root_dir=pathlib.Path.cwd())
    assert conn.container_args(ctx) == []


def test_connector_string_command():
    conn = ConnectorEnvironment(
        name="my-connector", image=Image.from_base("python:3.11"), command="python app.py --debug"
    )
    ctx = SerializationContext(org="org", project="proj", domain="dev", version="v1", root_dir=pathlib.Path.cwd())
    cmd = conn.container_cmd(ctx)
    assert cmd == ["python", "app.py", "--debug"]


def test_connector_list_command():
    conn = ConnectorEnvironment(
        name="my-connector", image=Image.from_base("python:3.11"), command=["python", "-m", "main"]
    )
    ctx = SerializationContext(org="org", project="proj", domain="dev", version="v1", root_dir=pathlib.Path.cwd())
    cmd = conn.container_cmd(ctx)
    assert cmd == ["python", "-m", "main"]


@pytest.mark.parametrize("reserved_port", [8012, 8022, 8112, 9090, 9091])
def test_connector_reserved_port(reserved_port):
    with pytest.raises(ValueError, match="is not allowed"):
        ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"), port=reserved_port)


def test_connector_invalid_name():
    with pytest.raises(ValueError, match="must consist of lower case"):
        ConnectorEnvironment(name="My_Connector", image=Image.from_base("python:3.11"))


def test_connector_lifecycle_decorators():
    conn = ConnectorEnvironment(name="my-connector", image=Image.from_base("python:3.11"))

    @conn.on_startup
    def startup():
        pass

    @conn.server
    def serve():
        pass

    @conn.on_shutdown
    def shutdown():
        pass

    assert conn._on_startup is startup
    assert conn._server is serve
    assert conn._on_shutdown is shutdown
