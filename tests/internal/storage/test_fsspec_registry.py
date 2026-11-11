"""
Tests for how flyte interacts with the global fsspec registry.

flyte v1 (flytekit) may share the process (e.g. a v1 task importing ``flyte`` to launch v2
runs) and registers s3fs/gcsfs/adlfs for the object-store protocols. Importing ``flyte`` must
never clobber those registrations, and flyte's own I/O must not depend on what is registered.
"""

import importlib.util
import sys

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from flyteidl2.core import literals_pb2, types_pb2
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.registry import _registry, known_implementations
from obstore.fsspec import FsspecStore

import flyte.storage as storage
from flyte.io import DataFrame
from flyte.io._dataframe import basic_dfs
from flyte.storage._storage import (
    _OBSTORE_SUPPORTED_PROTOCOLS,
    _obstore_filesystem_class,
    _register_obstore_for_missing_protocols,
    get_underlying_filesystem,
)


@pytest.fixture
def restore_fsspec_registry():
    """Snapshot the global fsspec registry and restore it after the test."""
    saved = dict(_registry)
    yield
    _registry.clear()
    _registry.update(saved)


class DummyS3FileSystem(fsspec.AbstractFileSystem):
    """Stand-in for s3fs.S3FileSystem: accepts s3fs-only kwargs like ``cache_regions``."""

    protocol = "s3"
    cachable = False

    def __init__(self, *args, cache_regions: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_regions = cache_regions


def test_pre_registered_implementation_survives(restore_fsspec_registry):
    """A registration made before flyte's helper runs (e.g. by flytekit) must be preserved."""
    fsspec.register_implementation("s3", DummyS3FileSystem, clobber=True)

    _register_obstore_for_missing_protocols()

    assert fsspec.registry["s3"] is DummyS3FileSystem
    assert isinstance(fsspec.filesystem("s3"), DummyS3FileSystem)


def test_get_underlying_filesystem_is_registry_independent(restore_fsspec_registry):
    """flyte's own I/O gets an obstore-backed filesystem even when s3fs-style owns the registry."""
    fsspec.register_implementation("s3", DummyS3FileSystem, clobber=True)

    fs = get_underlying_filesystem("s3")

    assert isinstance(fs, FsspecStore)
    assert fs.protocol == "s3"
    # The obstore bypasses in _storage.py duck-type on these attributes.
    assert hasattr(fs, "_split_path")
    assert hasattr(fs, "_construct_store")


def test_helper_registers_obstore_when_nothing_installed(restore_fsspec_registry, monkeypatch):
    """When no implementation is registered or installed, obstore is registered as a fallback."""
    for protocol in _OBSTORE_SUPPORTED_PROTOCOLS:
        _registry.pop(protocol, None)
        monkeypatch.setitem(known_implementations, protocol, {"class": "flyte_test_not_installed.FS", "err": ""})

    _register_obstore_for_missing_protocols()

    for protocol in _OBSTORE_SUPPORTED_PROTOCOLS:
        assert protocol in fsspec.registry
        assert issubclass(fsspec.registry[protocol], FsspecStore)


@pytest.fixture
def installed_s3_package(restore_fsspec_registry, monkeypatch, tmp_path):
    """
    An installed package that fsspec knows as the "s3" implementation but has not imported yet.
    Importing it raises, like an optional package with a broken dependency.
    """
    package = "flyte_test_installed_s3fs"
    (tmp_path / f"{package}.py").write_text("raise RuntimeError('broken optional dependency')\n")
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setitem(known_implementations, "s3", {"class": f"{package}.FS", "err": ""})
    _registry.pop("s3", None)
    yield package
    sys.modules.pop(package, None)


def test_helper_leaves_installed_implementation_alone(installed_s3_package):
    """
    fsspec resolves known_implementations lazily, so the registry may be empty at import time
    even though e.g. s3fs is installed. The helper must check for the installed package, not
    just the registry, and leave the protocol to it.
    """
    _register_obstore_for_missing_protocols()

    assert "s3" not in fsspec.registry


def test_helper_does_not_import_other_implementations(installed_s3_package):
    """
    The check must not import the other implementation: that would add s3fs/gcsfs/adlfs import
    time to `import flyte`, and a package that fails to import would fail `import flyte` too.
    """
    _register_obstore_for_missing_protocols()

    assert installed_s3_package not in sys.modules


def test_helper_registers_obstore_when_lookup_fails(restore_fsspec_registry, monkeypatch):
    """If looking the package up raises, fall back to obstore instead of failing the import."""
    _registry.pop("s3", None)

    def raise_value_error(name):
        raise ValueError(f"{name}.__spec__ is None")

    monkeypatch.setattr(importlib.util, "find_spec", raise_value_error)

    _register_obstore_for_missing_protocols()

    assert issubclass(fsspec.registry["s3"], FsspecStore)


def test_flytekit_style_kwargs_reach_registered_class(restore_fsspec_registry):
    """
    Simulate the flytekit collision: flytekit calls fsspec.filesystem("s3", cache_regions=True).
    With s3fs-style registered, those kwargs must reach that class instead of obstore (which
    rejects them with "Configuration key: 'cache_regions' is not valid for store 'S3'").
    """
    fsspec.register_implementation("s3", DummyS3FileSystem, clobber=True)
    _register_obstore_for_missing_protocols()

    fs = fsspec.filesystem("s3", cache_regions=True)

    assert isinstance(fs, DummyS3FileSystem)
    assert fs.cache_regions is True


def test_obstore_filesystem_class_mirrors_obstore_register():
    """The flyte-owned class matches what obstore.fsspec.register(asynchronous=True) creates."""
    cls = _obstore_filesystem_class("s3")

    assert issubclass(cls, FsspecStore)
    assert cls.protocol == "s3"
    assert cls.asynchronous is True
    # The per-protocol class is cached.
    assert _obstore_filesystem_class("s3") is cls
    # fsspec's _Cached metaclass instance caching still applies to direct instantiation.
    assert cls() is cls()


def test_flyte_protocol_registration_intact():
    """The `flyte` protocol registration is unaffected by the gap-filling helper."""
    from flyte.storage._remote_fs import FlyteFS

    assert fsspec.registry["flyte"] is FlyteFS


class RejectingS3FileSystem(fsspec.AbstractFileSystem):
    """Stand-in for s3fs/adlfs that fails the test if anything resolves "s3" or "abfs" through the registry."""

    protocol = "s3"
    cachable = False

    def __init__(self, *args, **kwargs):
        raise AssertionError(f"flyte resolved a protocol through the global fsspec registry with {kwargs}")


class InMemoryS3FileSystem(MemoryFileSystem):
    """Stand-in for flyte's own obstore-backed S3 and Azure filesystems."""

    protocol = "s3"
    cachable = False

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(path.split("://", 1)[-1])


@pytest.fixture
def flyte_owned_s3(restore_fsspec_registry, monkeypatch):
    """s3fs-style owns the registry; flyte's own filesystem is an in-memory stand-in."""
    fsspec.register_implementation("s3", RejectingS3FileSystem, clobber=True)
    fsspec.register_implementation("abfs", RejectingS3FileSystem, clobber=True)
    fs = InMemoryS3FileSystem()
    calls = []

    def fake_get_underlying_filesystem(protocol=None, anonymous=False, path=None, **kwargs):
        calls.append({"path": path, "anonymous": anonymous})
        return fs

    monkeypatch.setattr(storage, "get_underlying_filesystem", fake_get_underlying_filesystem)
    yield fs, calls
    MemoryFileSystem.store.clear()


AZURE_URI = "abfs://ctr@acct.dfs.core.windows.net/ds"


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", ["s3://bkt/ds", AZURE_URI])
async def test_pandas_parquet_handlers_do_not_use_registry(flyte_owned_s3, uri):
    """
    The pandas handlers must reach S3 and Azure through flyte's own filesystem. Handing flyte's obstore-shaped
    kwargs to pandas as storage_options resolves "s3" through the registry, and s3fs rejects them
    ("AioSession.__init__() got an unexpected keyword argument 'retry_config'").
    """
    fs, _ = flyte_owned_s3
    df = pd.DataFrame({"a": [1, 2, 3]})

    sd = await basic_dfs.PandasToParquetEncodingHandler().encode(
        DataFrame.from_df(df, uri=uri), types_pb2.StructuredDatasetType()
    )
    assert fs.exists(f"{fs._strip_protocol(uri)}/00000")

    out = await basic_dfs.ParquetToPandasDecodingHandler().decode(sd, literals_pb2.StructuredDatasetMetadata())
    pd.testing.assert_frame_equal(out, df)


@pytest.mark.asyncio
async def test_pandas_csv_handlers_do_not_use_registry(flyte_owned_s3, monkeypatch):
    fs, calls = flyte_owned_s3
    df = pd.DataFrame({"a": [1, 2, 3]})

    sd = await basic_dfs.PandasToCSVEncodingHandler().encode(
        DataFrame.from_df(df, uri="s3://bkt/csv"), types_pb2.StructuredDatasetType()
    )
    assert fs.cat_file("bkt/csv/data.csv") == b"a\n1\n2\n3\n"

    # The decoder reads through storage.get_stream and only falls back to pandas for anonymous access.
    class NoCredentialsError(Exception):
        pass

    async def no_credentials(path):
        raise NoCredentialsError
        yield b""

    monkeypatch.setattr(storage, "get_stream", no_credentials)

    out = await basic_dfs.CSVToPandasDecodingHandler().decode(sd, literals_pb2.StructuredDatasetMetadata())
    pd.testing.assert_frame_equal(out, df)
    assert calls[-1] == {"path": "s3://bkt/csv/data.csv", "anonymous": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", ["s3://bkt/arrow", AZURE_URI])
async def test_arrow_parquet_decoder_uses_flyte_filesystem(flyte_owned_s3, uri):
    fs, calls = flyte_owned_s3
    table = pa.table({"a": [1, 2, 3]})
    pq.write_table(table, f"{fs._strip_protocol(uri)}/00000", filesystem=fs)

    sd = literals_pb2.StructuredDataset(uri=uri)
    out = await basic_dfs.ParquetToArrowDecodingHandler().decode(sd, literals_pb2.StructuredDatasetMetadata())
    assert out.equals(table)
    assert calls[-1] == {"path": uri, "anonymous": False}


def test_pandas_handlers_leave_other_paths_to_pandas(tmp_path):
    local = str(tmp_path / "ds")
    assert basic_dfs._get_pandas_path_and_filesystem(local) == (local, None)
    assert basic_dfs._get_pandas_path_and_filesystem("gs://bkt/ds") == ("gs://bkt/ds", None)
