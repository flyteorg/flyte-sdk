"""`flyte.artifacts.Artifact` was a runtime_checkable protocol in 2.10.x; `isinstance` keeps that meaning."""

import warnings

import pytest

import flyte.artifacts as artifacts
from flyte.artifacts import Artifact, ArtifactLike
from flyte.artifacts._handle import is_handle
from flyte.io import File


class _Like:
    def get_artifact_metadata(self):
        return None


def test_handle_isinstance_is_plain():
    h = Artifact("compat_h", type=File)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(h, Artifact)
        assert isinstance(Artifact.ref("other", type=File), Artifact)
        assert not isinstance(3, Artifact)
    assert is_handle(h)


def test_artifact_like_isinstance_falls_back_with_deprecation():
    v = artifacts.new(File(path="s3://b/x"), artifacts.Metadata(name="compat"))
    with pytest.warns(DeprecationWarning, match="flyte.artifacts.ArtifactLike"):
        assert isinstance(v, Artifact)
    with pytest.warns(DeprecationWarning):
        assert isinstance(_Like(), Artifact)
    assert isinstance(_Like(), ArtifactLike)
    # Strict checks never take the fallback.
    assert not is_handle(v)
    assert not isinstance(_Like(), artifacts.ArtifactRef)


def test_subscript_still_a_type_hint():
    assert Artifact[File] is Artifact
