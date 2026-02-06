"""
Integration Tests for Flyte SDK Examples

This module contains integration tests for all runnable examples in the examples/ directory.
Tests are organized by category and use pytest markers for selective execution.

Test Execution Tiers:
  - Tier 1: Core tests (every PR)
    pytest -m "integration and not (gpu or external or stress or apps or secrets)"

  - Tier 2: Including apps (nightly)
    pytest -m "integration and not (gpu or external or stress or secrets)"

  - Tier 3: GPU tests (weekly, GPU runners)
    pytest -m "integration and gpu"

  - Tier 4: All tests (manual)
    pytest -m "integration"
"""

import logging
import os
from datetime import datetime, timedelta

import pytest

import flyte
from flyte._code_bundle import build_code_bundle

# =============================================================================
# FIXTURES
# =============================================================================


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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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
        print("  Completed successfully\n")


async def _deploy_and_verify(env_or_app, test_name: str):
    """
    Helper function to deploy an app and verify deployment succeeded.

    Args:
        env_or_app: The AppEnvironment or app to deploy
        test_name: Name of the test for logging purposes

    Returns:
        List of Deployment objects
    """
    deployments = await flyte.deploy.aio(env_or_app)
    print(f"\n[{test_name}]")
    for deployment in deployments:
        for env_name, deployed_env in deployment.envs.items():
            print(f"  Deployed: {deployed_env.get_name()}")
    print("  Deployment successful\n")
    return deployments


# =============================================================================
# BASICS - CORE EXAMPLES (existing tests)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_hello(flyte_client):
    """Test the basics.hello example with a list of integers."""
    from examples.basics.hello import main

    await _run_and_wait(main, "test_basics_hello", x_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


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


# =============================================================================
# BASICS - ADDITIONAL TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_map(flyte_client):
    """Test the basics.map example with map/reduce pattern."""
    from examples.basics.map import main

    await _run_and_wait(main, "test_basics_map", n=5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_enum_vals(flyte_client):
    """Test the basics.enum_vals example with enums and literals."""
    from examples.basics.enum_vals import main

    await _run_and_wait(main, "test_basics_enum_vals")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_frames(flyte_client):
    """Test the basics.frames example with pandas DataFrames."""
    from examples.basics.frames import main

    await _run_and_wait(main, "test_basics_frames")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_from_scratch(flyte_client):
    """Test the basics.from_scratch example."""
    from examples.basics.from_scratch import main

    await _run_and_wait(main, "test_basics_from_scratch", n=5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_no_outputs(flyte_client):
    """Test the basics.no_outputs example with void return."""
    from examples.basics.no_outputs import main

    await _run_and_wait(main, "test_basics_no_outputs")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_repeated_tasks(flyte_client):
    """Test the basics.repeated_tasks example with many task invocations."""
    from examples.basics.repeated_tasks import main_task

    await _run_and_wait(main_task, "test_basics_repeated_tasks")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_report_example(flyte_client):
    """Test the basics.report_example with Flyte reports."""
    from examples.basics.report_example import main

    await _run_and_wait(main, "test_basics_report_example")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_hello_polyglot(flyte_client):
    """Test the basics.hello_polyglot example with external package."""
    from examples.basics.hello_polyglot import main

    await _run_and_wait(main, "test_basics_hello_polyglot", letter="e")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_overwrite_existing_file(flyte_client):
    """Test the basics.overwrite_existing_file example."""
    from examples.basics.overwrite_existing_file import main

    await _run_and_wait(main, "test_basics_overwrite_existing_file")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basics_offloaded_type_basic_usage(flyte_client):
    """Test the basics.offloaded_type_basic_usage example."""
    from examples.basics.offloaded_type_basic_usage import main

    await _run_and_wait(main, "test_basics_offloaded_type_basic_usage")


# =============================================================================
# BASICS - TYPES
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_types_simple_types(flyte_client):
    """Test the basics.types.simple_types example."""
    from examples.basics.types.simple_types import main

    await _run_and_wait(
        main,
        "test_types_simple_types",
        str="World",
        int=42,
        float=3.14,
        bool=True,
        start_time=datetime.now(),
        duration=timedelta(hours=1),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_types_dataclass_types(flyte_client):
    """Test the basics.types.dataclass_types example with complex dataclass."""
    from examples.basics.types.dataclass_types import ComplexData, NestedData, main

    nested1 = NestedData(name="first", value=10, score=95.5)
    nested2 = NestedData(name="second", value=20, score=87.3)

    complex_data = ComplexData(
        str_field="Hello World!",
        int_field=42,
        float_field=3.14,
        bool_field=True,
        start_time=datetime.now(),
        duration=timedelta(hours=1),
        string_list=["hello", "world"],
        int_list=[1, 2, 3],
        float_list=[1.1, 2.2],
        bool_list=[True, False],
        nested=nested1,
        nested_list=[nested1, nested2],
        optional_str="optional",
        optional_int=99,
    )

    await _run_and_wait(main, "test_types_dataclass_types", data=complex_data)


# =============================================================================
# ADVANCED
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_advanced_display_names(flyte_client):
    """Test the advanced.display_names example with short names."""
    from examples.advanced.display_names import main

    await _run_and_wait(main, "test_advanced_display_names")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_advanced_custom_context(flyte_client):
    """Test the advanced.custom_context example."""
    from examples.advanced.custom_context import main

    await _run_and_wait(main, "test_advanced_custom_context", x=10)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_advanced_cancel_tasks(flyte_client):
    """Test the advanced.cancel_tasks example with task cancellation."""
    from examples.advanced.cancel_tasks import main

    await _run_and_wait(main, "test_advanced_cancel_tasks", n=3, f=2.0)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_advanced_leaky_coroutines(flyte_client):
    """Test the advanced.leaky_coroutines example."""
    from examples.advanced.leaky_coroutines import main

    await _run_and_wait(main, "test_advanced_leaky_coroutines", seconds=5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_advanced_local_tasks(flyte_client):
    """Test the advanced.local_tasks example with traces."""
    from examples.advanced.local_tasks import parallel_main_no_io

    await _run_and_wait(parallel_main_no_io, "test_advanced_local_tasks", q="hello")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_advanced_multi_loops(flyte_client):
    """Test the advanced.multi_loops example with memory retries."""
    from examples.advanced.multi_loops import main

    await _run_and_wait(main, "test_advanced_multi_loops", n=2)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_advanced_udfs(flyte_client):
    """Test the advanced.udfs example with user-defined functions."""
    from examples.advanced.udfs import main

    await _run_and_wait(main, "test_advanced_udfs")


# =============================================================================
# ML
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ml_model_training(flyte_client):
    """Test the ml.model_training example with full ML pipeline."""
    from examples.ml.model_training import main

    await _run_and_wait(main, "test_ml_model_training")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ml_rfe(flyte_client):
    """Test the ml.rfe (recursive feature elimination) example."""
    from examples.ml.rfe import rfe

    await _run_and_wait(rfe, "test_ml_rfe")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ml_distributed_random_forest(flyte_client):
    """Test the ml.distributed_random_forest example."""
    from examples.ml.distributed_random_forest import main

    await _run_and_wait(main, "test_ml_distributed_random_forest", n_estimators=4)


# =============================================================================
# REPORTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reports_streaming_reports(flyte_client):
    """Test the reports.streaming_reports example with shorter parameters."""
    from examples.reports.streaming_reports import training_loss_visualization

    await _run_and_wait(training_loss_visualization, "test_reports_streaming_reports", epochs=5)


# =============================================================================
# STREAMING
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_basic_as_completed(flyte_client):
    """Test the streaming.basic_as_completed example."""
    from examples.streaming.basic_as_completed import streaming_reduce_processing

    await _run_and_wait(streaming_reduce_processing, "test_streaming_basic_as_completed")


# =============================================================================
# SYNC
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sync_async_in_sync(flyte_client):
    """Test the sync.async_in_sync example."""
    from examples.sync.async_in_sync import call_async

    await _run_and_wait(call_async, "test_sync_async_in_sync")


# =============================================================================
# REUSE
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reuse_reusable(flyte_client):
    """Test the reuse.reusable example with reuse policy."""
    from examples.reuse.reusable import main

    await _run_and_wait(main, "test_reuse_reusable", n=10)


# =============================================================================
# CACHING
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_caching_content_based_caching(flyte_client):
    """Test the caching.content_based_caching example."""
    from examples.caching.content_based_caching import demo_cache_behavior

    await _run_and_wait(demo_cache_behavior, "test_caching_content_based_caching")


# =============================================================================
# GENAI
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_genai_hello_agent(flyte_client):
    """Test the genai.hello_agent example with simple agent pattern."""
    from examples.genai.hello_agent import ResearchState, lead_agent

    state = ResearchState(query="test query")
    await _run_and_wait(lead_agent, "test_genai_hello_agent", state=state, num_subagents=2)


# =============================================================================
# DATA PROCESSING
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_processing_deltalake(flyte_client):
    """Test the data_processing.deltalake_example with smaller dataset."""
    from examples.data_processing.deltalake_example import main

    await _run_and_wait(main, "test_data_processing_deltalake", rows=100)


# =============================================================================
# PLUGINS (existing tests)
# =============================================================================


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


# =============================================================================
# APPS (deploy-based tests)
# =============================================================================


@pytest.mark.integration
@pytest.mark.apps
@pytest.mark.asyncio
async def test_apps_basic_app(flyte_client):
    """Test the apps.basic_app example with Streamlit."""
    from examples.apps.basic_app import app_env

    await _deploy_and_verify(app_env, "test_apps_basic_app")


@pytest.mark.integration
@pytest.mark.apps
@pytest.mark.asyncio
async def test_apps_single_script_fastapi(flyte_client):
    """Test the apps.single_script_fastapi example."""
    from examples.apps.single_script_fastapi import env

    await _deploy_and_verify(env, "test_apps_single_script_fastapi")


# =============================================================================
# GPU TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_accelerators_gpu(flyte_client):
    """Test GPU accelerator example."""
    from examples.accelerators.gpu import main

    await _run_and_wait(main, "test_accelerators_gpu")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_genai_vllm_app(flyte_client):
    """Test vLLM app deployment."""
    from examples.genai.vllm.vllm_app import vllm_app

    await _deploy_and_verify(vllm_app, "test_genai_vllm_app")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_genai_sglang_app(flyte_client):
    """Test SGLang app deployment."""
    from examples.genai.sglang.sglang_app import sglang_app

    await _deploy_and_verify(sglang_app, "test_genai_sglang_app")


# =============================================================================
# EXTERNAL SERVICE TESTS (skipped - require backend connector configuration)
# =============================================================================


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.skip(reason="Requires BigQuery connector configured in Union backend")
@pytest.mark.asyncio
async def test_connectors_bigquery(flyte_client):
    """Test BigQuery connector."""
    from examples.connectors.bigquery_example import bigquery_task

    await _run_and_wait(bigquery_task, "test_connectors_bigquery")


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.skip(reason="Requires Databricks connector configured in Union backend")
@pytest.mark.asyncio
async def test_connectors_databricks(flyte_client):
    """Test Databricks connector."""
    from examples.connectors.databricks_example import hello_databricks_nested

    await _run_and_wait(hello_databricks_nested, "test_connectors_databricks")


@pytest.mark.integration
@pytest.mark.secrets
@pytest.mark.skip(reason="Requires W&B API key configured as secret in Union backend")
@pytest.mark.asyncio
async def test_context_wandb(flyte_client):
    """Test W&B context."""
    from examples.context.wandb_context import main

    await _run_and_wait(main, "test_context_wandb")


# =============================================================================
# STRESS TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_large_fanout(flyte_client):
    """Test large fanout stress test."""
    from examples.stress.large_fanout import main

    await _run_and_wait(main, "test_stress_large_fanout")


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_runs_per_second(flyte_client):
    """Test runs-per-second stress test."""
    from examples.stress.runs_per_second import main

    await _run_and_wait(main, "test_stress_runs_per_second")


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_large_file_io(flyte_client):
    """Test large file I/O stress test."""
    from examples.stress.large_file_io import main

    await _run_and_wait(main, "test_stress_large_file_io")


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_fanout_concurrency(flyte_client):
    """Test fanout concurrency stress test."""
    from examples.stress.fanout_concurrency import reuse_concurrency

    await _run_and_wait(reuse_concurrency, "test_stress_fanout_concurrency")
