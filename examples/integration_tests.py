import os

import pytest

import flyte


@pytest.fixture(scope="session")
def flyte_client():
    """
    Initialize Flyte client once for all tests.
    """
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

    yield flyte


async def _run_and_wait(flyte_client, task_fn, test_name: str, **kwargs):
    """
    Helper function to run a Flyte task and wait for completion.

    Args:
        flyte_client: The Flyte client fixture
        task_fn: The task function to run
        test_name: Name of the test for logging purposes
        **kwargs: Keyword arguments to pass to the task

    Raises:
        Any exception raised by run.wait() will propagate
    """
    run = await flyte.run.aio(task_fn, **kwargs)

    print(f"\n[{test_name}]")
    print(f"  Run name: {run.name}")
    print(f"  Run URL: {run.url}")

    run.wait()

    print("  ✓ Completed successfully\n")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_hello(flyte_client):
    """Test the basics.hello example with a list of integers."""
    from examples.basics.hello import main

    await _run_and_wait(flyte_client, main, "test_basics_hello", x_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_spark(flyte_client):
    """Test the Spark plugin example."""
    from examples.plugins.spark_example import hello_spark_nested

    await _run_and_wait(flyte_client, hello_spark_nested, "test_spark")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ray(flyte_client):
    """Test the Ray plugin example."""
    from examples.plugins.ray_example import hello_ray_nested

    await _run_and_wait(flyte_client, hello_ray_nested, "test_ray")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dask(flyte_client):
    """Test the Dask plugin example."""
    from examples.plugins.dask_example import hello_dask_nested

    await _run_and_wait(flyte_client, hello_dask_nested, "test_dask")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pytorch(flyte_client):
    """Test the PyTorch plugin example."""
    from examples.plugins.torch_example import torch_distributed_train

    await _run_and_wait(flyte_client, torch_distributed_train, "test_pytorch", epochs=1)
