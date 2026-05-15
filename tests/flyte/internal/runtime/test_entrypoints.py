import os
import pathlib
import tempfile

from flyte._internal.runtime import entrypoints
from flyte._internal.runtime.entrypoints import _list_user_files, _resolve_bundle_destination
from flyte.models import CodeBundle


def test_list_user_files_excludes_venv():
    """Test that _list_user_files filters out venv directories."""
    with tempfile.TemporaryDirectory() as cwd:
        # Create user files
        os.makedirs(os.path.join(cwd, "mypackage"), exist_ok=True)
        open(os.path.join(cwd, "mypackage", "main.py"), "w").close()
        open(os.path.join(cwd, "mypackage", "__init__.py"), "w").close()

        # Create venv files that should be excluded
        os.makedirs(os.path.join(cwd, "venv", "lib", "python3.12", "site-packages"), exist_ok=True)
        open(os.path.join(cwd, "venv", "lib", "python3.12", "site-packages", "pkg.py"), "w").close()

        # Create .venv files that should be excluded
        os.makedirs(os.path.join(cwd, ".venv", "lib"), exist_ok=True)
        open(os.path.join(cwd, ".venv", "lib", "something.py"), "w").close()

        files = _list_user_files(cwd)

        assert os.path.join("mypackage", "main.py") in files
        assert os.path.join("mypackage", "__init__.py") in files

        # venv and .venv files should not appear
        for f in files:
            assert not f.startswith("venv" + os.sep), f"venv file should be excluded: {f}"
            assert not f.startswith(".venv" + os.sep), f".venv file should be excluded: {f}"


def test_list_user_files_excludes_uv_and_local():
    """Test that _list_user_files filters out .local and .uv directories."""
    with tempfile.TemporaryDirectory() as cwd:
        # Create user file
        open(os.path.join(cwd, "hello.py"), "w").close()

        # Create .local directory (uv installs Python here)
        os.makedirs(os.path.join(cwd, ".local", "share", "uv", "python"), exist_ok=True)
        open(os.path.join(cwd, ".local", "share", "uv", "python", "cpython.py"), "w").close()

        # Create .uv directory
        os.makedirs(os.path.join(cwd, ".uv", "cache"), exist_ok=True)
        open(os.path.join(cwd, ".uv", "cache", "cached.whl"), "w").close()

        files = _list_user_files(cwd)

        assert "hello.py" in files

        for f in files:
            assert not f.startswith(".local" + os.sep), f".local file should be excluded: {f}"
            assert not f.startswith(".uv" + os.sep), f".uv file should be excluded: {f}"


def test_list_user_files_excludes_pycache_and_git():
    """Test that _list_user_files filters out __pycache__ and .git directories."""
    with tempfile.TemporaryDirectory() as cwd:
        open(os.path.join(cwd, "app.py"), "w").close()

        os.makedirs(os.path.join(cwd, "__pycache__"), exist_ok=True)
        open(os.path.join(cwd, "__pycache__", "app.cpython-312.pyc"), "w").close()

        os.makedirs(os.path.join(cwd, ".git", "objects"), exist_ok=True)
        open(os.path.join(cwd, ".git", "objects", "abc123"), "w").close()

        files = _list_user_files(cwd)

        assert "app.py" in files

        for f in files:
            assert not f.startswith("__pycache__" + os.sep), f"__pycache__ file should be excluded: {f}"
            assert not f.startswith(".git" + os.sep), f".git file should be excluded: {f}"


def test_list_user_files_excludes_site_packages():
    """Test that _list_user_files filters out site-packages even when nested."""
    with tempfile.TemporaryDirectory() as cwd:
        open(os.path.join(cwd, "workflow.py"), "w").close()

        # site-packages nested inside a lib directory
        os.makedirs(os.path.join(cwd, "lib", "site-packages", "numpy"), exist_ok=True)
        open(os.path.join(cwd, "lib", "site-packages", "numpy", "__init__.py"), "w").close()

        files = _list_user_files(cwd)

        assert "workflow.py" in files

        for f in files:
            assert "site-packages" not in f, f"site-packages file should be excluded: {f}"


def test_list_user_files_only_lists_files_not_dirs():
    """Test that _list_user_files only returns files, not directory entries."""
    with tempfile.TemporaryDirectory() as cwd:
        os.makedirs(os.path.join(cwd, "mypackage"), exist_ok=True)
        open(os.path.join(cwd, "mypackage", "main.py"), "w").close()

        files = _list_user_files(cwd)

        # Should contain files only, not bare directory names
        assert os.path.join("mypackage", "main.py") in files
        assert "mypackage" not in files


def test_list_user_files_empty_directory():
    """Test that _list_user_files returns empty list for empty directory."""
    with tempfile.TemporaryDirectory() as cwd:
        files = _list_user_files(cwd)
        assert files == []


def _bundle(destination: str, version: str = "v0") -> CodeBundle:
    return CodeBundle(tgz="s3://bucket/code.tgz", destination=destination, computed_version=version)


def test_resolve_bundle_destination_keeps_writable_cwd(tmp_path, monkeypatch):
    """When CWD is writable+traversable, destination is unchanged."""
    monkeypatch.chdir(tmp_path)
    bundle = _bundle(destination=".")
    assert _resolve_bundle_destination(bundle) is bundle


def test_resolve_bundle_destination_keeps_explicit_destination(monkeypatch):
    """An explicit destination is honored verbatim, even if it would also be unwritable."""
    monkeypatch.setattr(os, "access", lambda *a, **kw: False)
    bundle = _bundle(destination="/some/explicit/path")
    assert _resolve_bundle_destination(bundle) is bundle


def test_resolve_bundle_destination_falls_back_when_cwd_not_writable(monkeypatch, tmp_path):
    """Default '.' destination is rerouted under /tmp when CWD is not writable."""
    fake_root = tmp_path / "flyte-bundle"
    monkeypatch.setattr(entrypoints, "_BUNDLE_FALLBACK_ROOT", fake_root)
    monkeypatch.setattr(os, "access", lambda path, mode: False if path == "." else True)
    bundle = _bundle(destination=".", version="abc123")
    resolved = _resolve_bundle_destination(bundle)
    assert resolved is not bundle
    assert resolved.destination == str(fake_root / "abc123")
    assert pathlib.Path(resolved.destination).is_dir()
    # Other fields are preserved.
    assert resolved.tgz == bundle.tgz
    assert resolved.computed_version == bundle.computed_version


def test_resolve_bundle_destination_handles_empty_destination(monkeypatch, tmp_path):
    """An empty-string destination is treated like '.' and triggers the fallback."""
    fake_root = tmp_path / "flyte-bundle"
    monkeypatch.setattr(entrypoints, "_BUNDLE_FALLBACK_ROOT", fake_root)
    monkeypatch.setattr(os, "access", lambda *a, **kw: False)
    bundle = _bundle(destination="", version="def456")
    resolved = _resolve_bundle_destination(bundle)
    assert resolved.destination == str(fake_root / "def456")
