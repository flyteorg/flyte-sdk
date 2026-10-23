import errno
import gzip
import pathlib
import subprocess

import pytest

from flyte import _sentry
from flyte._code_bundle import _packaging
from flyte._code_bundle._packaging import _compress_tarball


@pytest.fixture
def source(tmp_path: pathlib.Path) -> pathlib.Path:
    src = tmp_path / "tmp.tar"
    src.write_bytes(b"code bundle contents " * 4096)
    return src


def _fail_pigz(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    monkeypatch.setattr(_packaging.shutil, "which", lambda name: "/usr/bin/pigz")

    def _run(cmd, stdout=None, check=False, **kwargs):
        # Simulate pigz writing a partial stream before dying.
        if stdout is not None:
            stdout.write(b"partial garbage")
        raise exc

    monkeypatch.setattr(_packaging.subprocess, "run", _run)


def test_compress_tarball_falls_back_to_gzip_when_pigz_fails(monkeypatch, source, tmp_path):
    """FLYTE-SDK-8W: pigz exiting non-zero (28 = ENOSPC) must not crash the deploy; gzip takes over."""
    _fail_pigz(monkeypatch, subprocess.CalledProcessError(28, ["/usr/bin/pigz", "--no-time", "-c", str(source)]))
    output = tmp_path / "out.tar.gz"

    _compress_tarball(source, output)

    assert gzip.decompress(output.read_bytes()) == source.read_bytes()


def test_compress_tarball_falls_back_to_gzip_when_pigz_cannot_launch(monkeypatch, source, tmp_path):
    _fail_pigz(monkeypatch, PermissionError(errno.EACCES, "Permission denied", "/usr/bin/pigz"))
    output = tmp_path / "out.tar.gz"

    _compress_tarball(source, output)

    assert gzip.decompress(output.read_bytes()) == source.read_bytes()


def test_compress_tarball_disk_full_surfaces_as_user_environment_oserror(monkeypatch, source, tmp_path):
    """When the disk really is full, the gzip fallback raises OSError(ENOSPC), which Sentry filters."""
    _fail_pigz(monkeypatch, subprocess.CalledProcessError(28, ["/usr/bin/pigz"]))

    class _FullDiskGzipFile(gzip.GzipFile):
        def write(self, data):
            raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(_packaging.gzip, "GzipFile", _FullDiskGzipFile)

    with pytest.raises(OSError) as exc_info:
        _compress_tarball(source, tmp_path / "out.tar.gz")

    assert exc_info.value.errno == errno.ENOSPC
    assert _sentry._is_user_error(exc_info.value)
