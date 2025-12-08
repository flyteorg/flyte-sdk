from __future__ import annotations

import pytest

from flyte.models import PathRewrite


def test_path_rewrite():
    with pytest.raises(ValueError):
        PathRewrite.from_str("s3://old_prefix:/tmp/new_prefix")

    with pytest.raises(ValueError):
        PathRewrite.from_str("s3://old_prefix-/tmp/new_prefix")

    with pytest.raises(ValueError):
        PathRewrite.from_str("s3://old_prefix/tmp/new_prefix")

    with pytest.raises(ValueError):
        PathRewrite.from_str("s3://old_prefix- >/tmp/new_prefix")

    pr = PathRewrite.from_str("s3://old_prefix/->/tmp/new_prefix/")
    assert pr.old_prefix == "s3://old_prefix/"
    assert pr.new_prefix == "/tmp/new_prefix/"
