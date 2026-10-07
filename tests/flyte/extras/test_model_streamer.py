"""ModelStreamer against a real (tiny) transformers checkpoint in a local object store."""

import pathlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from obstore.store import LocalStore  # noqa: E402

from flyte.extras.model_streamer import (  # noqa: E402
    ModelStreamer,
    empty_weights,
    load_hf_model,
    missing_parameters,
)

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
    """A tied-embedding Llama, sharded so the index file is exercised."""
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


def _reference(model):
    return {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}


def test_checkpoint_is_sharded(store):
    root, _ = store
    files = sorted(p.name for p in (root / "bucket/models/tiny").iterdir())
    assert "model.safetensors.index.json" in files
    assert sum(f.endswith(".safetensors") for f in files) > 1


@pytest.mark.asyncio
async def test_stream_yields_every_tensor_inside_a_running_loop(store):
    _, model = store
    got = await ModelStreamer(REMOTE, chunk_size=1024, max_concurrency=4).state_dict()
    ref = _reference(model)
    assert got.keys() == ref.keys()
    for name, tensor in ref.items():
        assert torch.equal(got[name], tensor), name


@pytest.mark.asyncio
async def test_stream_casts_floating_dtype(store):
    got = await ModelStreamer(REMOTE).state_dict(device="cpu", dtype=torch.bfloat16)
    assert {t.dtype for t in got.values()} == {torch.bfloat16}


@pytest.mark.asyncio
async def test_stream_sync_works_on_a_thread_with_a_running_loop(store):
    # flyte's own get_tensors() raises here ("cannot be called from a running event loop").
    _, model = store
    got = dict(ModelStreamer(REMOTE, chunk_size=1024).stream_sync())
    assert got.keys() == _reference(model).keys()


def test_stream_sync_propagates_errors(store):
    with pytest.raises(ValueError, match="No files found"):
        list(ModelStreamer("s3://bucket/models/absent").stream_sync())


def test_stream_sync_stops_cleanly_when_abandoned(store):
    it = ModelStreamer(REMOTE, chunk_size=1024, max_concurrency=1).stream_sync()
    next(it)
    it.close()  # joins the background thread; would hang if the producer could not stop


@pytest.mark.asyncio
async def test_load_into_meta_module(store):
    _, model = store
    with empty_weights():
        empty = transformers.LlamaForCausalLM(model.config)
    assert missing_parameters(empty)
    inv_freq = empty.model.rotary_emb.inv_freq
    assert inv_freq.device.type == "cpu"  # buffers are built for real

    result = await ModelStreamer(REMOTE).load_into(empty, device="cpu")
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert empty.lm_head.weight is empty.model.embed_tokens.weight  # re-tied
    for name, tensor in model.state_dict().items():
        assert torch.equal(empty.state_dict()[name], tensor), name


@pytest.mark.asyncio
async def test_load_into_strict_reports_mismatch(store):
    _, model = store
    with empty_weights():
        empty = transformers.LlamaForCausalLM(model.config)
    with pytest.raises(KeyError, match="unexpected"):
        await ModelStreamer(REMOTE).load_into(empty, device="cpu", key_mapping=lambda k: "x." + k)

    result = await ModelStreamer(REMOTE).load_into(
        empty, device="cpu", strict=False, key_mapping=lambda k: None if "layers.1." in k else k
    )
    assert result.missing_keys and all("layers.1." in k for k in result.missing_keys)


@pytest.mark.asyncio
async def test_load_hf_model_matches_from_pretrained(store, tmp_path):
    _, model = store
    loaded, local = await load_hf_model(REMOTE, device="cpu", local_dir=tmp_path / "meta")
    assert (local / "config.json").exists()
    assert not list(local.glob("*.safetensors"))  # weights were streamed, not downloaded
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        assert torch.allclose(loaded(ids).logits, model(ids).logits)
    assert not missing_parameters(loaded)


def test_stream_sync_without_a_running_loop(store):
    _, model = store
    assert dict(ModelStreamer(REMOTE).stream_sync()).keys() == _reference(model).keys()


@pytest.mark.asyncio
async def test_load_hf_model_reconciles_base_model_prefix(store, tmp_path):
    # The checkpoint was saved from LlamaForCausalLM ("model.layers..."); loading the bare
    # LlamaModel ("layers...") needs the prefix stripped, as from_pretrained does.
    _, model = store
    loaded, _ = await load_hf_model(
        REMOTE, device="cpu", model_class=transformers.AutoModel, local_dir=tmp_path / "meta"
    )
    assert isinstance(loaded, transformers.LlamaModel)
    assert not missing_parameters(loaded)
    assert torch.equal(loaded.layers[0].mlp.up_proj.weight, model.model.layers[0].mlp.up_proj.weight)


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork()")
def test_a_forked_child_fails_fast_instead_of_hanging(store):
    # obstore's runtime does not survive fork(): a forked child's first read would block forever.
    import os
    import posix  # the child exits through posix._exit: tests/conftest.py mocks os._exit

    _, model = store
    assert dict(ModelStreamer(REMOTE).stream_sync()).keys() == _reference(model).keys()

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        os.close(read_fd)
        try:
            list(ModelStreamer(REMOTE).stream_sync())
            message = b"no error"
        except RuntimeError as exc:
            message = str(exc).encode()
        os.write(write_fd, message)
        posix._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as f:
        message = f.read().decode()
    os.waitpid(pid, 0)
    assert "forked" in message and "spawn" in message
