import pathlib

import pytest

from flyte.connectors import _connector
from flyte.models import CodeBundle


@pytest.mark.asyncio
async def test_upload_code_bundle_cleans_up_local_copy(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle_dirs: list[pathlib.Path] = []

    async def build_code_bundle(
        *, from_dir: pathlib.Path, dryrun: bool, copy_bundle_to: pathlib.Path | None = None
    ) -> CodeBundle:
        assert from_dir == tmp_path
        assert dryrun is True
        assert copy_bundle_to is not None
        bundle_dirs.append(copy_bundle_to)
        local_bundle = copy_bundle_to / "bundle.tar.gz"
        local_bundle.write_bytes(b"bundle")
        return CodeBundle(tgz=str(local_bundle), computed_version="v1")

    async def put(from_path: str, to_path: str) -> str:
        local_path = pathlib.Path(from_path)
        assert bundle_dirs
        assert local_path.parent == bundle_dirs[0]
        assert local_path.exists()
        assert to_path == "s3://bucket/prefix/code_bundle/bundle.tar.gz"
        return "s3://bucket/uploaded/bundle.tar.gz"

    monkeypatch.setattr(_connector, "build_code_bundle", build_code_bundle)
    monkeypatch.setattr(_connector.storage, "put", put)

    result = await _connector._build_and_upload_code_bundle(tmp_path, "s3://bucket/prefix")

    assert result == CodeBundle(
        tgz="s3://bucket/uploaded/bundle.tar.gz",
        computed_version="v1",
        destination="/opt/flyte/",
    )
    assert len(bundle_dirs) == 1
    assert not bundle_dirs[0].exists()


@pytest.mark.asyncio
async def test_upload_code_bundle_cleans_up_local_copy_when_upload_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dirs: list[pathlib.Path] = []

    async def build_code_bundle(
        *, from_dir: pathlib.Path, dryrun: bool, copy_bundle_to: pathlib.Path | None = None
    ) -> CodeBundle:
        assert from_dir == tmp_path
        assert dryrun is True
        assert copy_bundle_to is not None
        bundle_dirs.append(copy_bundle_to)
        local_bundle = copy_bundle_to / "bundle.tar.gz"
        local_bundle.write_bytes(b"bundle")
        return CodeBundle(tgz=str(local_bundle), computed_version="v1")

    async def put(from_path: str, to_path: str) -> str:
        local_path = pathlib.Path(from_path)
        assert bundle_dirs
        assert local_path.parent == bundle_dirs[0]
        assert local_path.exists()
        assert to_path == "s3://bucket/prefix/code_bundle/bundle.tar.gz"
        raise OSError("upload failed")

    monkeypatch.setattr(_connector, "build_code_bundle", build_code_bundle)
    monkeypatch.setattr(_connector.storage, "put", put)

    with pytest.raises(OSError, match="upload failed"):
        await _connector._build_and_upload_code_bundle(tmp_path, "s3://bucket/prefix")

    assert len(bundle_dirs) == 1
    assert not bundle_dirs[0].exists()
