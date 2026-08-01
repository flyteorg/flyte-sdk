"""Regression test for the Card.create_from temp-file flush bug: the upload
must see the fully flushed content, not an empty unflushed file."""

import pathlib
from unittest.mock import AsyncMock, patch

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
