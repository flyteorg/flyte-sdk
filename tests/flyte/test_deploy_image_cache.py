import sys
from unittest.mock import AsyncMock, patch

import pytest

import flyte
from flyte._deploy import DeploymentPlan, _build_images
from flyte._image import Image
from flyte._task_environment import TaskEnvironment


@pytest.mark.parametrize(
    "python_version,expected_py_version",
    [
        (None, "{}.{}".format(sys.version_info.major, sys.version_info.minor)),  # Use local python version
        ((3, 10), "3.10"),
    ],
)
@pytest.mark.asyncio
async def test_create_image_cache_lookup(python_version, expected_py_version):
    """Test that _build_images creates the correct nested dictionary structure for ImageCache."""

    if python_version is None:
        mock_image = Image.from_debian_base().with_pip_packages("numpy")
    else:
        mock_image = Image.from_debian_base(python_version=python_version).with_pip_packages("numpy")

    env_name = "test_env"
    fake_image_uri = f"registry.example.com/test-py{expected_py_version}:latest"

    mock_env = TaskEnvironment(name=env_name, image=mock_image)
    deployment_plan = DeploymentPlan(envs={env_name: mock_env})

    with patch("flyte._deploy._build_image_bg", new_callable=AsyncMock) as mock_build:
        mock_build.return_value = (env_name, fake_image_uri, None)

        image_cache = await _build_images(deployment_plan)

        # Check that environment name is present in the image_lookup dict
        assert env_name in image_cache.image_lookup

        # Check the image_lookup dict contains the expected image URI
        assert image_cache.image_lookup[env_name] == fake_image_uri


# ---------------------------------------------------------------------------
# seed_cache: nested runs reuse URIs already resolved by the launching run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_images_seed_cache_reuses_uri_and_skips_build():
    """An env present in seed_cache reuses the recorded URI; the builder is never invoked."""
    flyte.init()
    from flyte._internal.imagebuild.image_builder import ImageCache

    env_name = "seeded_env"
    seeded_uri = "356633062068.dkr.ecr.us-east-2.amazonaws.com/union/demo:flyte-abc123"
    env = TaskEnvironment(name=env_name, image=Image.from_debian_base(python_version=(3, 12)))
    plan = DeploymentPlan(envs={env_name: env})
    seed = ImageCache(image_lookup={env_name: seeded_uri})

    with patch("flyte._deploy._build_image_bg", new_callable=AsyncMock) as mock_build:
        image_cache = await _build_images(plan, seed_cache=seed)

    mock_build.assert_not_called()
    assert image_cache.image_lookup[env_name] == seeded_uri


@pytest.mark.asyncio
async def test_build_images_seed_cache_partial_hit():
    """Envs missing from seed_cache still build; seeded envs are reused."""
    flyte.init()
    from flyte._internal.imagebuild.image_builder import ImageCache

    seeded_uri = "registry.example.com/prebuilt:tag1"
    built_uri = "registry.example.com/fresh:tag2"
    seeded_env = TaskEnvironment(name="seeded_env", image=Image.from_debian_base(python_version=(3, 12)))
    fresh_env = TaskEnvironment(name="fresh_env", image=Image.from_debian_base(python_version=(3, 12)))
    plan = DeploymentPlan(envs={"seeded_env": seeded_env, "fresh_env": fresh_env})
    seed = ImageCache(image_lookup={"seeded_env": seeded_uri})

    with patch("flyte._deploy._build_image_bg", new_callable=AsyncMock) as mock_build:
        mock_build.return_value = ("fresh_env", built_uri, None)
        image_cache = await _build_images(plan, seed_cache=seed)

    assert mock_build.call_count == 1
    assert mock_build.call_args[0][0] == "fresh_env"
    assert image_cache.image_lookup == {"seeded_env": seeded_uri, "fresh_env": built_uri}


@pytest.mark.asyncio
async def test_build_images_no_seed_builds_everything():
    """seed_cache=None preserves existing behavior."""
    flyte.init()
    env_name = "plain_env"
    built_uri = "registry.example.com/fresh:tag3"
    env = TaskEnvironment(name=env_name, image=Image.from_debian_base(python_version=(3, 12)))
    plan = DeploymentPlan(envs={env_name: env})

    with patch("flyte._deploy._build_image_bg", new_callable=AsyncMock) as mock_build:
        mock_build.return_value = (env_name, built_uri, None)
        image_cache = await _build_images(plan, seed_cache=None)

    assert mock_build.call_count == 1
    assert image_cache.image_lookup[env_name] == built_uri


def test_ambient_image_cache_none_on_driver():
    """Outside a task pod there is no task context, so no seed is used."""
    from flyte._run import _ambient_image_cache

    assert _ambient_image_cache() is None


def test_ambient_image_cache_returns_transported_cache_in_pod():
    """Inside a task pod the transported compiled_image_cache is returned as the seed."""
    from unittest.mock import Mock

    from flyte._internal.imagebuild.image_builder import ImageCache
    from flyte._run import _ambient_image_cache

    cache = ImageCache(image_lookup={"env_a": "registry.example.com/img:tag"})
    fake_ctx = Mock()
    fake_ctx.data.task_context.compiled_image_cache = cache

    with patch("flyte._run.internal_ctx", return_value=fake_ctx):
        assert _ambient_image_cache() is cache
