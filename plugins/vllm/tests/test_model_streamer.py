"""The flyte-streaming vLLM load format against a tiny checkpoint in a local object store."""

import asyncio
import pathlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from obstore.store import LocalStore  # noqa: E402

from flyteplugins.vllm import model_streamer  # noqa: E402

REMOTE = "s3://bucket/models/tiny"


class _LocalObstoreFS:
    """The two private hooks flyte's streamer calls on its obstore-backed fsspec filesystem."""

    def __init__(self, root: pathlib.Path):
        self.root = root

    def _split_path(self, path: str):
        bucket, _, prefix = path.removeprefix("s3://").partition("/")
        return bucket, prefix

    def _construct_store(self, bucket: str):
        return LocalStore(str(self.root / bucket))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    model.save_pretrained(root / "bucket/models/tiny", max_shard_size="20KB")
    return root, model


@pytest.fixture
def store(checkpoint, monkeypatch):
    root, model = checkpoint
    fs = _LocalObstoreFS(root)
    import flyte.app.extras._model_loader.loader as loader
    import flyte.storage._storage as storage

    monkeypatch.setattr(loader, "get_underlying_filesystem", lambda **_: fs)
    monkeypatch.setattr(storage, "get_underlying_filesystem", lambda **_: fs)
    return root, model


@pytest.mark.asyncio
async def test_engine_args(store, tmp_path):
    args = await model_streamer.engine_args(REMOTE, local_dir=tmp_path / "meta", max_concurrency=8)
    assert args["load_format"] == model_streamer.LOAD_FORMAT == "flyte-streaming"
    assert args["model"] == str(tmp_path / "meta")
    assert args["model_loader_extra_config"]["flyte_remote_path"] == REMOTE
    assert args["model_loader_extra_config"]["flyte_max_concurrency"] == 8
    assert (tmp_path / "meta/config.json").exists()
    assert not list((tmp_path / "meta").glob("*.safetensors"))


def test_register_is_a_noop_without_vllm():
    try:
        import vllm  # noqa: F401
    except ImportError:
        model_streamer.register()  # must not raise
    else:
        pytest.skip("vllm installed; covered by test_loader_streams_through_vllm")


def test_loader_streams_through_vllm(store):
    pytest.importorskip("vllm")
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER, get_model_loader
    from vllm.plugins import load_general_plugins

    _, model = store
    # vLLM's own plugin discovery registers the format through the entry point.
    load_general_plugins()
    model_streamer.register()  # also covers an editable install without entry-point metadata
    assert model_streamer.LOAD_FORMAT in _LOAD_FORMAT_TO_MODEL_LOADER

    args = asyncio.run(model_streamer.engine_args(REMOTE))
    config = LoadConfig(
        load_format=args["load_format"],
        # vLLM's default loader rejects keys it does not know; ours must be stripped first.
        model_loader_extra_config={**args["model_loader_extra_config"], "enable_multithread_load": False},
    )
    loader = get_model_loader(config)
    assert loader.load_config.model_loader_extra_config == {"enable_multithread_load": False}
    assert config.model_loader_extra_config["flyte_remote_path"] == REMOTE  # caller's config untouched

    class _Model:
        pass

    async def inside_a_running_loop():
        return dict(loader.get_all_weights(None, _Model()))

    got = asyncio.run(inside_a_running_loop())
    want = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    assert got.keys() == want.keys()
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name

    with pytest.raises(ValueError, match="flyte_remote_path"):
        get_model_loader(LoadConfig(load_format=model_streamer.LOAD_FORMAT))
