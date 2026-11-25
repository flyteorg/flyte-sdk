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
async def test_engine_args(store, tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    args = await model_streamer.engine_args(REMOTE, local_dir=tmp_path / "meta", max_concurrency=8)
    assert args["load_format"] == model_streamer.LOAD_FORMAT == "flyte-streaming"
    assert args["model"] == str(tmp_path / "meta")
    assert args["model_loader_extra_config"]["flyte_remote_path"] == REMOTE
    assert args["model_loader_extra_config"]["flyte_max_concurrency"] == 8
    assert (tmp_path / "meta/config.json").exists()
    assert not list((tmp_path / "meta").glob("*.safetensors"))
    # The engine must not fork from a process that has used obstore.
    import os

    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


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


# ---------------------------------------------------------------------------
# Node cache: one download per node, shared by its tensor-parallel ranks
# ---------------------------------------------------------------------------


def _want(model):
    return {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}


@pytest.mark.asyncio
async def test_engine_args_passes_node_cache_dir(store, tmp_path):
    args = await model_streamer.engine_args(REMOTE, local_dir=tmp_path / "meta", node_cache_dir=tmp_path / "cache")
    assert args["model_loader_extra_config"]["flyte_node_cache_dir"] == str(tmp_path / "cache")
    plain = await model_streamer.engine_args(REMOTE, local_dir=tmp_path / "meta2")
    assert "flyte_node_cache_dir" not in plain["model_loader_extra_config"]


def test_node_cache_downloads_once_for_all_ranks(store, tmp_path, monkeypatch):
    """Ranks that load at the same time share one download and get every tensor."""
    import threading

    from flyte.extras.model_streamer import ModelStreamer

    _, model = store
    downloads = []
    real_download = ModelStreamer.download

    async def counting_download(self, local_dir):
        downloads.append(local_dir)
        return await real_download(self, local_dir)

    monkeypatch.setattr(ModelStreamer, "download", counting_download)

    cache = tmp_path / "cache"
    results: list[dict] = []
    errors: list[BaseException] = []

    def rank():
        try:
            streamer = ModelStreamer(REMOTE)
            results.append(dict(model_streamer._weights_from_node_cache(streamer, cache)))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=rank) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(downloads) == 1
    want = _want(model)
    assert len(results) == 4
    for got in results:
        assert got.keys() == want.keys()
        for name, tensor in want.items():
            assert torch.equal(got[name], tensor), name

    # A later engine on the same node reuses the copy.
    dict(model_streamer._weights_from_node_cache(ModelStreamer(REMOTE), cache))
    assert len(downloads) == 1


def test_node_cache_rejects_another_checkpoint(store, tmp_path):
    from flyte.extras.model_streamer import ModelStreamer

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ".flyte-weights-complete").write_text("s3://bucket/models/other")
    with pytest.raises(ValueError, match="already holds the weights of s3://bucket/models/other"):
        dict(model_streamer._weights_from_node_cache(ModelStreamer(REMOTE), cache))


def test_node_cache_ignores_a_crashed_download_of_another_checkpoint(store, tmp_path):
    """Shards left by an unfinished download are removed, not loaded with the new weights."""
    from safetensors.torch import save_file

    from flyte.extras.model_streamer import ModelStreamer

    _, model = store
    cache = tmp_path / "cache"
    cache.mkdir()
    # No completion marker: another checkpoint's download died partway.
    save_file({"stale.weight": torch.ones(2)}, str(cache / "model-00009-of-00009.safetensors"))
    save_file({"model.norm.weight": torch.zeros(32)}, str(cache / "model-00001-of-00002.safetensors"))

    got = dict(model_streamer._weights_from_node_cache(ModelStreamer(REMOTE), cache))

    want = _want(model)
    assert got.keys() == want.keys()
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name
    assert not (cache / "model-00009-of-00009.safetensors").exists()


def test_node_cache_reads_only_indexed_tensors(store, tmp_path):
    """With an index, a tensor or file it doesn't list is never loaded."""
    from safetensors.torch import save_file

    from flyte.extras.model_streamer import ModelStreamer

    _, model = store
    cache = tmp_path / "cache"
    dict(model_streamer._weights_from_node_cache(ModelStreamer(REMOTE), cache))
    save_file({"extra.weight": torch.ones(2)}, str(cache / "zz-extra.safetensors"))

    got = dict(model_streamer._weights_from_node_cache(ModelStreamer(REMOTE), cache))
    assert got.keys() == _want(model).keys()


def test_loader_reads_node_cache_through_vllm(store, tmp_path):
    pytest.importorskip("vllm")
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader

    _, model = store
    model_streamer.register()
    args = asyncio.run(model_streamer.engine_args(REMOTE, node_cache_dir=tmp_path / "cache"))
    loader = get_model_loader(
        LoadConfig(load_format=args["load_format"], model_loader_extra_config=args["model_loader_extra_config"])
    )

    class _Model:
        pass

    got = dict(loader.get_all_weights(None, _Model()))
    want = _want(model)
    assert got.keys() == want.keys()
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name
    assert (tmp_path / "cache/.flyte-weights-complete").read_text() == REMOTE
