import flyte
from flyte._doc import Documentation as _Documentation
from flyte._doc import extract_docstring


def test_documentation_is_exported():
    assert flyte.Documentation is _Documentation
    assert "Documentation" in flyte.__all__


def test_extract_docstring():
    def f():
        """Trains a model."""

    assert extract_docstring(f) == flyte.Documentation(description="Trains a model.")
    assert extract_docstring(None) == flyte.Documentation(description="")
