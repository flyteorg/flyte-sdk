import pytest

from flyte._image import _BASE_REGISTRY
from flyte._internal.imagebuild.remote_builder import _custom_repository


@pytest.mark.parametrize(
    "repository, expected",
    [
        # Bare name (no registry) -> default repo, no override.
        ("ai", ""),
        # Multi-segment name without a registry host -> default repo, no override.
        ("union/demo", ""),
        # _BASE_REGISTRY (default imagebuilder repo) -> default repo, no override.
        (f"{_BASE_REGISTRY}/ai", ""),
        # Custom ECR registry (clone(registry=...)) -> passed through so GetImage looks there (SE-775).
        (
            "869913524964.dkr.ecr.us-west-2.amazonaws.com/ai",
            "869913524964.dkr.ecr.us-west-2.amazonaws.com/ai",
        ),
        # Custom registry with a nested repo path -> preserved.
        (
            "869913524964.dkr.ecr.us-west-2.amazonaws.com/custom/repo",
            "869913524964.dkr.ecr.us-west-2.amazonaws.com/custom/repo",
        ),
    ],
)
def test_custom_repository(repository, expected):
    assert _custom_repository(repository) == expected
