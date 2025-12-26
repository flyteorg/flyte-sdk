import logging
import os

import pytest

import flyte
from flyte._code_bundle import build_code_bundle


@pytest.fixture(autouse=True)
def clear_lru_caches():
    build_code_bundle.cache_clear()


@pytest.fixture(scope="session")
def flyte_client():
    """
    Initialize Flyte client once for all tests.
    """
    if os.getenv("GITHUB_ACTIONS", "") == "true":
        flyte.init(
            endpoint=os.getenv("FLYTE_ENDPOINT", "dns:///playground.canary.unionai.cloud"),
            auth_type="ClientSecret",
            client_id="flyte-sdk-ci",
            client_credentials_secret=os.getenv("FLYTE_SDK_CI_TOKEN"),
            insecure=False,
            image_builder="remote",
            project=os.getenv("FLYTE_PROJECT", "flyte-sdk"),
            domain=os.getenv("FLYTE_DOMAIN", "development"),
        )
    else:
        flyte.init(
            endpoint=os.getenv("FLYTE_ENDPOINT", "dns:///playground.canary.unionai.cloud"),
            auth_type="Pkce",
            insecure=False,
            image_builder="remote",
            project=os.getenv("FLYTE_PROJECT", "flyte-sdk"),
            domain=os.getenv("FLYTE_DOMAIN", "development"),
        )

    yield flyte


async def _run_and_wait(task_fn, test_name: str, **kwargs):
    """
    Helper function to run a Flyte task and wait for completion.

    Args:
        task_fn: The task function to run
        test_name: Name of the test for logging purposes
        **kwargs: Keyword arguments to pass to the task

    Raises:
        Any exception raised by run.wait() will propagate
    """
    run = await flyte.with_runcontext(log_level=logging.DEBUG).run.aio(task_fn, **kwargs)

    print(f"\n[{test_name}]")
    print(f"  Run name: {run.name}")
    print(f"  Run URL: {run.url}")

    run.wait()
    detail = await run.action.details()
    if detail.error_info:
        raise RuntimeError(f"Run failed with error: {detail.error_info.message}")
    else:
        print("  ✓ Completed successfully\n")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_hello(flyte_client):
    """Test the basics.hello example with a list of integers."""
    from examples.basics.hello import main

    await _run_and_wait(main, "test_basics_hello", x_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_spark(flyte_client):
    """Test the Spark plugin example."""
    from examples.plugins.spark_example import hello_spark_nested

    await _run_and_wait(hello_spark_nested, "test_spark")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ray(flyte_client):
    """Test the Ray plugin example."""
    from examples.plugins.ray_example import hello_ray_nested

    await _run_and_wait(hello_ray_nested, "test_ray")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dask(flyte_client):
    """Test the Dask plugin example."""
    from examples.plugins.dask_example import hello_dask_nested

    await _run_and_wait(hello_dask_nested, "test_dask")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pytorch(flyte_client):
    """Test the PyTorch plugin example."""
    from examples.plugins.torch_example import torch_distributed_train

    await _run_and_wait(torch_distributed_train, "test_pytorch", epochs=1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_flyte_file(flyte_client):
    """Test the Flyte File async API example."""
    from examples.basics.file import main

    await _run_and_wait(main, "test_flyte_file")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_flyte_directory(flyte_client):
    """Test the Flyte Directory async API example."""
    from examples.basics.dir import main

    await _run_and_wait(main, "test_flyte_directory")
