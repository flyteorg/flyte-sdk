import hashlib
import os
from pathlib import Path

from flyte._utils.file_handling import update_hasher_for_source


def _build_tree(root: Path):
    root.mkdir(parents=True)
    (root / "alpha").mkdir()
    (root / "alpha" / "nested").mkdir()
    (root / "alpha" / "a.txt").write_text("a")
    (root / "alpha" / "nested" / "n.txt").write_text("n")
    (root / "beta").mkdir()
    (root / "beta" / "b.txt").write_text("b")
    (root / "top.txt").write_text("top")


def _ordered_walk(reverse: bool):
    # A deterministic os.walk stand-in that visits entries in name order, so the
    # digest cannot depend on the host filesystem's readdir order.
    def walk(top, topdown=True, onerror=None, followlinks=False):
        def recurse(root):
            with os.scandir(root) as entries:
                names = sorted((e.name for e in entries), reverse=reverse)
            dirnames = [n for n in names if os.path.isdir(os.path.join(root, n))]
            files = [n for n in names if not os.path.isdir(os.path.join(root, n))]
            yield root, dirnames, files
            for dirname in dirnames:
                yield from recurse(os.path.join(root, dirname))

        yield from recurse(str(top))

    return walk


def _digest(source, walk) -> str:
    hasher = hashlib.sha256()
    real_walk = os.walk
    os.walk = walk
    try:
        update_hasher_for_source(source, hasher)
    finally:
        os.walk = real_walk
    return hasher.hexdigest()


def test_hash_independent_of_directory_traversal_order(tmp_path):
    tree = tmp_path / "tree"
    _build_tree(tree)

    ascending = _digest(tree, _ordered_walk(reverse=False))
    descending = _digest(tree, _ordered_walk(reverse=True))

    assert ascending == descending
