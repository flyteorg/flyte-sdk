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


# ---------------------------------------------------------------------------
# Cross-environment image resolution: independent envs are planned separately,
# but must still share one image cache so a task can call into a sibling env.
# ---------------------------------------------------------------------------


CLI_IMAGE = "my.registry.example.com/custom:v1"


@pytest.mark.asyncio
async def test_build_images_for_plans_merges_cli_image_across_independent_envs():
    """`flyte deploy --image <uri>` must reach every env, not just the first plan's.

    Independent environments (no `depends_on` link) each land in their own DeploymentPlan, so a
    per-plan image cache only ever knows about its own env.
    """
    from flyte._deploy import _build_images_for_plans, plan_deploy

    flyte.init()
    parent = TaskEnvironment(name="xenv-parent")
    sibling = TaskEnvironment(name="xenv-sibling", resources=flyte.Resources(cpu="0.5"))
    cloned = parent.clone_with(name="xenv-cloned", image="auto")

    plans = plan_deploy(parent, sibling, cloned)
    assert [list(p.envs) for p in plans] == [["xenv-parent"], ["xenv-sibling"], ["xenv-cloned"]]

    merged = await _build_images_for_plans(plans, {"default": CLI_IMAGE})

    assert merged.image_lookup == {
        "xenv-parent": CLI_IMAGE,
        "xenv-sibling": CLI_IMAGE,
        "xenv-cloned": CLI_IMAGE,
    }


@pytest.mark.asyncio
async def test_deploy_gives_every_plan_the_merged_image_cache():
    """Every deployed env's SerializationContext sees all envs deployed in the same invocation.

    The parent's container args carry this cache into the pod, and a child task in another env has
    its image resolved from it at runtime. Before this, the sibling env was absent from the
    parent's cache and silently fell back to the default flyteorg image.
    """
    import pathlib
    from unittest.mock import Mock

    from flyte._internal.runtime.task_serde import lookup_image_in_cache

    flyte.init()
    parent = TaskEnvironment(name="merged-parent")
    sibling = TaskEnvironment(name="merged-sibling", resources=flyte.Resources(cpu="0.5"))

    @sibling.task
    async def child(x: int) -> int:
        return x

    @parent.task
    async def root(x: int) -> int:
        return await child(x=x)

    fake_bundle = Mock()
    fake_bundle.computed_version = "bundle-v1"

    fake_cfg = Mock()
    fake_cfg.root_dir = pathlib.Path("/tmp")
    fake_cfg.images = {"default": CLI_IMAGE}
    fake_cfg.project, fake_cfg.domain, fake_cfg.org = "p", "d", "o"

    seen = []

    def _fake_get_deployer(_env_type):
        async def _deployer(context):
            seen.append(context.serialization_context)
            deployed = Mock()
            deployed.get_name.return_value = context.environment.name
            return deployed

        return _deployer

    with (
        patch("flyte._initialize.is_initialized", return_value=True),
        patch("flyte._deploy.get_init_config", return_value=fake_cfg),
        patch("flyte._code_bundle._includes.collect_env_include_files", return_value=[]),
        patch("flyte._code_bundle.build_code_bundle", new=AsyncMock(return_value=fake_bundle)),
        patch("flyte._deployer.get_deployer", new=_fake_get_deployer),
    ):
        await flyte.deploy.aio(parent, sibling, dry_run=True)

    # Two independent envs -> two plans -> two deployer invocations, both with the merged cache.
    assert len(seen) == 2
    for sc in seen:
        assert sc.image_cache.image_lookup == {
            "merged-parent": CLI_IMAGE,
            "merged-sibling": CLI_IMAGE,
        }

    # The parent's context can now resolve the sibling env's task image to the --image override
    # instead of silently falling back to the default flyteorg image.
    assert lookup_image_in_cache(seen[0], child.parent_env_name, child.image) == CLI_IMAGE
