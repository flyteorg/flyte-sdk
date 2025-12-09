"""Tests for pod_spec_from_resources GPU resource key configuration."""

from flyte._resources import Resources, pod_spec_from_resources


def test_pod_spec_from_resources_default_key(monkeypatch):
    """Test that default GPU resource key is used when no env var is set."""
    # Ensure FLYTE_K8S_GPU_RESOURCE_KEY is not set
    monkeypatch.delenv("FLYTE_K8S_GPU_RESOURCE_KEY", raising=False)

    # Call pod_spec_from_resources with GPU resources
    pod_spec = pod_spec_from_resources(requests=Resources(cpu="1", memory="1Gi", gpu=1))

    # Verify the pod spec structure
    assert pod_spec is not None
    assert len(pod_spec.containers) == 1

    # Get the primary container's resource requests
    container = pod_spec.containers[0]
    assert container.resources is not None
    assert container.resources.requests is not None

    # Convert to dict to safely check keys
    requests = dict(container.resources.requests)

    # Assert the default GPU key is used
    assert "nvidia.com/gpu" in requests
    assert requests["nvidia.com/gpu"] == 1


def test_pod_spec_from_resources_env_override(monkeypatch):
    """Test that environment variable overrides the default GPU resource key."""
    # Set the environment variable
    monkeypatch.setenv("FLYTE_K8S_GPU_RESOURCE_KEY", "custom.com/gpu")

    # Call pod_spec_from_resources with GPU resources
    pod_spec = pod_spec_from_resources(requests=Resources(cpu="1", memory="1Gi", gpu=1))

    # Verify the pod spec structure
    assert pod_spec is not None
    assert len(pod_spec.containers) == 1

    # Get the primary container's resource requests
    container = pod_spec.containers[0]
    assert container.resources is not None
    assert container.resources.requests is not None

    # Convert to dict to safely check keys
    requests = dict(container.resources.requests)

    # Assert the custom GPU key from env var is used
    assert "custom.com/gpu" in requests
    assert requests["custom.com/gpu"] == 1
    # Assert the default key is NOT present
    assert "nvidia.com/gpu" not in requests


def test_pod_spec_from_resources_explicit_overrides_env(monkeypatch):
    """Test that explicit parameter overrides both env var and default."""
    # Set the environment variable
    monkeypatch.setenv("FLYTE_K8S_GPU_RESOURCE_KEY", "custom.com/gpu")

    # Call pod_spec_from_resources with explicit GPU resource key
    pod_spec = pod_spec_from_resources(
        requests=Resources(cpu="1", memory="1Gi", gpu=1),
        k8s_gpu_resource_key="explicit.com/gpu"
    )

    # Verify the pod spec structure
    assert pod_spec is not None
    assert len(pod_spec.containers) == 1

    # Get the primary container's resource requests
    container = pod_spec.containers[0]
    assert container.resources is not None
    assert container.resources.requests is not None

    # Convert to dict to safely check keys
    requests = dict(container.resources.requests)

    # Assert the explicit GPU key is used
    assert "explicit.com/gpu" in requests
    assert requests["explicit.com/gpu"] == 1
    # Assert neither env var nor default key is present
    assert "custom.com/gpu" not in requests
    assert "nvidia.com/gpu" not in requests
