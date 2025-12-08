"""
Updated pytest-based integration tests replacing the old integration_tests.py

This file shows how to migrate from the old asyncio-based integration tests
to pytest-based tests for better test isolation and reporting.

Migration from old integration_tests.py:
- Old: Manual async function with asyncio.gather()
- New: Pytest fixtures and parametrized tests

Usage:
    pytest examples/integration_tests_pytest.py -v
"""

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_hello(flyte_client):
    from examples.basics.hello import main
    run = await flyte.run.aio(main, x_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print(f"\nRun name: {run.name}")
    print(f"Run URL: {run.url}")

    run.wait()

    print(f"✓ Test completed successfully")
