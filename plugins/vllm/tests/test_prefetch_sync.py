"""Guards the version contract between prefetch's sharder and this plugin's model loader.

`flyte.prefetch.hf_model(shard_config=...)` shards a checkpoint by calling vLLM's
`save_sharded_state`, which writes rank-partitioned tensors under names taken from *that*
vLLM's model implementation. `FlyteModelLoader._load_sharded_model` in this plugin reads them
back into a `state_dict` built from *its* vLLM's model implementation and raises
`Missing keys ... in loaded state!` for anything it cannot fill.

Nothing at either end declares the other's version, so the two pins can drift apart in a
single-sided bump and the mismatch only surfaces when a real deploy loads a real sharded
checkpoint -- after an image build, a GPU allocation, and a weight download. These tests turn
that into a CI failure instead.
"""

from flyte.prefetch._hf_model import VLLM_SHARDING_IMAGE_PACKAGES, VLLM_SHARDING_VERSION

from flyteplugins.vllm._constants import VLLM_MIN_VERSION_STR


def test_prefetch_sharding_version_matches_plugin_pin():
    assert VLLM_SHARDING_VERSION == VLLM_MIN_VERSION_STR, (
        f"prefetch shards with vllm=={VLLM_SHARDING_VERSION} but this plugin serves with "
        f"vllm=={VLLM_MIN_VERSION_STR}. Sharded state written by one is not guaranteed to "
        f"load in the other -- bump flyte.prefetch._hf_model.VLLM_SHARDING_VERSION and "
        f"flyteplugins.vllm._constants.VLLM_MIN_VERSION together."
    )


def test_prefetch_sharding_image_installs_the_plugin_pin():
    """The constant is only worth asserting if the image actually installs it."""
    assert f"vllm=={VLLM_MIN_VERSION_STR}" in VLLM_SHARDING_IMAGE_PACKAGES
