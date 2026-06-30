import flyte
from flyte._build import ImageBuild


def test_image_build_dataclass():
    ib = ImageBuild(uri="docker.io/my-image:latest", remote_run=None)
    assert ib.uri == "docker.io/my-image:latest"
    assert ib.remote_run is None


def test_image_build_none_uri():
    ib = ImageBuild(uri=None, remote_run=None)
    assert ib.uri is None


def test_image_build_importable():
    assert flyte.ImageBuild is ImageBuild
