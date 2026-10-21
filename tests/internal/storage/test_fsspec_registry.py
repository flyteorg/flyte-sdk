"""
Tests for how flyte interacts with the global fsspec registry.

flyte v1 (flytekit) may share the process (e.g. a v1 task importing ``flyte`` to launch v2
runs) and registers s3fs/gcsfs/adlfs for the object-store protocols. Importing ``flyte`` must
never clobber those registrations, and flyte's own I/O must not depend on what is registered.
"""

import fsspec
import pytest
from fsspec.registry import _registry
from obstore.fsspec import FsspecStore

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


def test_helper_registers_obstore_when_nothing_importable(restore_fsspec_registry, monkeypatch):
    """When no implementation is registered or importable, obstore is registered as a fallback."""

    def raise_import_error(protocol):
        raise ImportError(f"no implementation for {protocol}")

    monkeypatch.setattr(fsspec, "get_filesystem_class", raise_import_error)
    for protocol in _OBSTORE_SUPPORTED_PROTOCOLS:
        _registry.pop(protocol, None)

    _register_obstore_for_missing_protocols()

    for protocol in _OBSTORE_SUPPORTED_PROTOCOLS:
        assert protocol in fsspec.registry
        assert issubclass(fsspec.registry[protocol], FsspecStore)


def test_helper_leaves_importable_implementation_alone(restore_fsspec_registry, monkeypatch):
    """
    fsspec resolves known_implementations lazily, so the registry may be empty at import time
    even though e.g. s3fs is installed. The helper must probe importability, not just the
    registry, and leave the importable implementation in place.
    """
    _registry.pop("s3", None)

    def fake_get_filesystem_class(protocol):
        if protocol == "s3":
            # Mimic fsspec's lazy resolution side effect of registering the class.
            fsspec.register_implementation("s3", DummyS3FileSystem, clobber=True)
            return DummyS3FileSystem
        raise ImportError(f"no implementation for {protocol}")

    monkeypatch.setattr(fsspec, "get_filesystem_class", fake_get_filesystem_class)

    _register_obstore_for_missing_protocols()

    assert fsspec.registry["s3"] is DummyS3FileSystem


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
