"""`flyte.remote` resolves its public names on first access instead of importing every entity."""

import subprocess
import sys

import pytest

import flyte.remote


def test_every_public_name_has_a_location():
    assert sorted(flyte.remote._EXPORTS) == sorted(flyte.remote.__all__)


@pytest.mark.parametrize("name", flyte.remote.__all__)
def test_public_name_resolves(name):
    assert getattr(flyte.remote, name) is not None
    assert name in dir(flyte.remote)


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no_such_entity"):
        flyte.remote.no_such_entity


def test_importing_the_client_does_not_import_the_entities():
    """The task runtime imports the client on every start; it must not drag the entities along."""
    code = (
        "import sys, flyte.remote._client.controlplane\n"
        "loaded = [m for m in sys.modules if m.startswith('flyte.remote._') and '_client' not in m]\n"
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
