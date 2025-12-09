"""
Example demonstrating the use of k8s_resource_key for GPU devices.

This example shows three ways to specify GPU resources:
1. Using explicit k8s_resource_key for custom resource names
2. Using partition to auto-synthesize MIG resource keys
3. Using default behavior (backward compatible)
"""

import flyte

# Example 1: Explicit k8s_resource_key
# This allows you to specify any Kubernetes GPU resource name directly
gpu_with_explicit_key = flyte.GPU(device="A100", quantity=1, k8s_resource_key="nvidia.com/mig-1g.10gb")
explicit_env = flyte.TaskEnvironment("explicit-mig", resources=flyte.Resources(gpu=gpu_with_explicit_key))


@explicit_env.task
async def task_with_explicit_mig() -> str:
    """Task requesting a specific MIG partition via explicit k8s_resource_key."""
    return "Running on nvidia.com/mig-1g.10gb"


# Example 2: Auto-synthesized MIG key from partition
# When a partition is specified without explicit k8s_resource_key,
# the system automatically synthesizes "nvidia.com/mig-{partition}"
gpu_with_partition = flyte.GPU(device="A100", quantity=1, partition="2g.10gb")
partition_env = flyte.TaskEnvironment("auto-mig", resources=flyte.Resources(gpu=gpu_with_partition))


@partition_env.task
async def task_with_auto_mig() -> str:
    """Task with auto-synthesized MIG key from partition (nvidia.com/mig-2g.10gb)."""
    return "Running on auto-synthesized nvidia.com/mig-2g.10gb"


# Example 3: Default behavior (backward compatible)
# When no partition or k8s_resource_key is specified, uses "nvidia.com/gpu"
gpu_default = flyte.GPU(device="T4", quantity=1)
default_env = flyte.TaskEnvironment("default-gpu", resources=flyte.Resources(gpu=gpu_default))


@default_env.task
async def task_with_default_gpu() -> str:
    """Task using default GPU resource key (nvidia.com/gpu)."""
    return "Running on nvidia.com/gpu"


# Example 4: Override priority - explicit key takes precedence over partition
# Even though a partition is specified, the explicit key is used
gpu_override = flyte.GPU(device="A100", quantity=1, partition="1g.5gb", k8s_resource_key="custom.com/gpu-resource")
override_env = flyte.TaskEnvironment("override-key", resources=flyte.Resources(gpu=gpu_override))


@override_env.task
async def task_with_override() -> str:
    """Task where explicit k8s_resource_key overrides partition-based synthesis."""
    return "Running on custom.com/gpu-resource"


if __name__ == "__main__":
    flyte.init_from_config()
    
    # Run the tasks to demonstrate different GPU resource configurations
    result1 = flyte.run(task_with_explicit_mig)
    print(f"Task 1: {result1.url}")
    
    result2 = flyte.run(task_with_auto_mig)
    print(f"Task 2: {result2.url}")
    
    result3 = flyte.run(task_with_default_gpu)
    print(f"Task 3: {result3.url}")
    
    result4 = flyte.run(task_with_override)
    print(f"Task 4: {result4.url}")
