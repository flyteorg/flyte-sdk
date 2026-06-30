import flyte


def test_version_returns_string():
    v = flyte.version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_version_matches_dunder():
    assert flyte.version() == flyte.__version__
