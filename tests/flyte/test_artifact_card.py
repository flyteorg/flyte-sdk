"""Regression test for the Card.create_from temp-file flush bug: the upload
must see the fully flushed content, not an empty unflushed file."""

import pathlib
from unittest.mock import patch

import pytest

from flyte.artifacts._card import Card


@pytest.mark.asyncio
async def test_create_from_content_uploads_flushed_file():
    captured = {}

    async def fake_upload(local_path: pathlib.Path, format="html", card_type="generic"):
        captured["content"] = local_path.read_text()
        return Card(uri="s3://b/card.html", format=format, card_type=card_type)

    with patch("flyte.artifacts._card._upload_card_from_local", side_effect=fake_upload):
        card = await Card.create_from.aio(content="<h1>Model Card</h1>", format="html", card_type="model")

    assert captured["content"] == "<h1>Model Card</h1>"
    assert card.uri == "s3://b/card.html"


@pytest.mark.asyncio
async def test_in_task_upload_sets_content_type():
    """Cards must be stored with their format's MIME type so presigned URLs
    render inline in the browser (binary/octet-stream blanks the iframe)."""
    from unittest.mock import MagicMock

    captured = {}

    async def fake_put_stream(data, *, to_path=None, **kwargs):
        captured["data"] = data
        captured["to_path"] = to_path
        captured["attributes"] = kwargs.get("attributes")
        return to_path

    fake_ctx = MagicMock()
    fake_ctx.output_path = "s3://bucket/meta"

    with (
        patch("flyte.ctx", return_value=fake_ctx),
        patch("flyte.storage.put_stream", side_effect=fake_put_stream),
    ):
        card = await Card.create_from.aio(content="<h1>c</h1>", format="html", card_type="model")

    assert captured["data"] == b"<h1>c</h1>"
    assert captured["to_path"] == "s3://bucket/meta/model.html"
    assert captured["attributes"]["Content-Type"] == "text/html"
    assert card.uri == "s3://bucket/meta/model.html"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fmt, expected",
    [("html", "text/html"), ("md", "text/markdown"), ("png", "image/png")],
)
async def test_local_upload_sets_content_type(fmt: str, expected: str):
    """Same requirement outside a task (the `flyte create artifact --card` path), where
    the card goes through the control plane's signed URL rather than the object store."""
    from unittest.mock import AsyncMock

    upload = AsyncMock(return_value=("md5", f"s3://bucket/card.{fmt}"))

    with (
        patch("flyte.ctx", return_value=None),
        patch("flyte.remote.upload_file.aio", upload),
    ):
        card = await Card.create_from.aio(content="hello", format=fmt, card_type="model")

    assert upload.await_args.kwargs["content_type"] == expected
    assert card.uri == f"s3://bucket/card.{fmt}"
