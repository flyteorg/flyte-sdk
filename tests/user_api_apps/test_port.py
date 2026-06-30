import pytest

from flyte.app import Port


def test_port_basic():
    p = Port(port=8080)
    assert p.port == 8080
    assert p.name is None


def test_port_with_name():
    p = Port(port=8080, name="h2c")
    assert p.port == 8080
    assert p.name == "h2c"


def test_port_frozen():
    p = Port(port=8080)
    with pytest.raises(AttributeError):
        p.port = 9090


def test_port_equality():
    p1 = Port(port=8080, name="h2c")
    p2 = Port(port=8080, name="h2c")
    assert p1 == p2


def test_port_inequality():
    p1 = Port(port=8080)
    p2 = Port(port=3000)
    assert p1 != p2


@pytest.mark.parametrize("reserved_port", [8012, 8022, 8112, 9090, 9091])
def test_port_reserved_ports_rejected(reserved_port):
    with pytest.raises(ValueError, match="is not allowed"):
        Port(port=reserved_port)


def test_port_valid_non_reserved():
    for p in [80, 443, 3000, 5000, 8080, 8081, 8500, 9000]:
        port = Port(port=p)
        assert port.port == p
