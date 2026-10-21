"""Unit tests for the DeepInfer clustered (multi-node) integration."""

import pytest
from flyte.clustered import JaxRun, TorchRun

from flyteplugins.deepinfer import clustered_deepinfer_environment


def test_clustered_env_defaults():
    env = clustered_deepinfer_environment(
        nodes=4,
        image="registry.io/deepinfer:sm100",
        gpu_per_node=8,
    )
    assert env.replicas == 4
    assert env.nproc_per_node == 1
    assert isinstance(env.runtime, JaxRun)
    assert env.image == "registry.io/deepinfer:sm100"
    assert env.resources is not None
    assert env.resources.gpu == 8


def test_clustered_env_no_resources():
    env = clustered_deepinfer_environment(nodes=2, image="reg/img:tag")
    assert env.resources is None


def test_clustered_env_runtime_override():
    env = clustered_deepinfer_environment(nodes=2, image="reg/img:tag", runtime=TorchRun())
    assert isinstance(env.runtime, TorchRun)
    assert env.nproc_per_node == 1


def test_clustered_env_rejects_zero_nodes():
    with pytest.raises(ValueError):
        clustered_deepinfer_environment(nodes=0, image="reg/img:tag")


def test_clustered_env_rejects_multi_nproc():
    with pytest.raises(ValueError):
        clustered_deepinfer_environment(nodes=2, image="reg/img:tag", nproc_per_node=2)
