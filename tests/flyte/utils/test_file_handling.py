import hashlib
import os

from flyte._utils import file_handling
from flyte._utils.file_handling import update_hasher_for_source


def _digest(source) -> str:
    hasher = hashlib.md5()
    update_hasher_for_source(source, hasher)
    return hasher.hexdigest()


def test_directory_digest_does_not_depend_on_listing_order(tmp_path, monkeypatch):
    for sub in ("alpha", "beta", "gamma"):
        (tmp_path / sub / "nested").mkdir(parents=True)
        (tmp_path / sub / "mod.py").write_text(f"# {sub}\n")
        (tmp_path / sub / "nested" / "leaf.py").write_text(f"# {sub} leaf\n")
    (tmp_path / "main.py").write_text("print('hi')\n")

    expected = _digest(tmp_path)

    real_walk = os.walk

    def walk_in_reverse_listing_order(*args, **kwargs):
        # The same entries, listed in another order -- as ext4 (hash order) or APFS may return them.
        for root, dirnames, files in real_walk(*args, **kwargs):
            dirnames.reverse()
            files.reverse()
            yield root, dirnames, files

    monkeypatch.setattr(file_handling.os, "walk", walk_in_reverse_listing_order)

    assert _digest(tmp_path) == expected
