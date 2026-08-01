from __future__ import annotations

import pathlib
import tempfile
from dataclasses import dataclass
from typing import Literal

import flyte
from flyte import storage, syncify

CardType = Literal["model", "data", "generic"]
CardFormat = Literal["html", "md", "json", "yaml", "csv", "tsv", "png", "jpg", "jpeg"]


@dataclass(frozen=True, kw_only=True)
class Card(object):
    uri: str
    format: CardFormat = "html"
    card_type: CardType = "generic"

    @syncify.syncify
    @classmethod
    async def create_from(
        cls,
        *,
        content: str | None = None,
        local_path: pathlib.Path | None = None,
        format: CardFormat = "html",
        card_type: CardType = "generic",
    ) -> Card:
        """
        Upload a card either from raw content or from a local file path.

        :param content: Raw content of the card to be uploaded.
        :param local_path: Local file path of the card to be uploaded.
        :param format: Format of the card (e.g., 'html', 'md',
                         'json', 'yaml', 'csv', 'tsv', 'png', 'jpg', 'jpeg').
        :param card_type: Type of the card (e.g., 'model', 'data', 'generic').
        """
        if content:
            # Close (and thereby flush) the temp file before uploading — reading
            # it inside the with block would see an empty, unflushed file.
            with tempfile.NamedTemporaryFile(mode="w", suffix=f".{format}", delete=False) as temp_file:
                temp_file.write(content)
                temp_path = pathlib.Path(temp_file.name)
            return await _upload_card_from_local(temp_path, format=format, card_type=card_type)
        if local_path:
            return await _upload_card_from_local(local_path, format=format, card_type=card_type)
        raise ValueError("Either content or local_path must be provided to upload a card.")


# MIME type per card format, stored as object metadata on upload so browsers
# render presigned card URLs inline (same mechanism as report uploads).
_FORMAT_CONTENT_TYPES: dict[str, str] = {
    "html": "text/html",
    "md": "text/markdown",
    "json": "application/json",
    "yaml": "application/yaml",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


async def _upload_card_from_local(
    local_path: pathlib.Path, format: CardFormat = "html", card_type: CardType = "generic"
) -> Card:
    # Implement upload. If in task context, upload to current metadata location, if not, upload using control plane.
    uri = ""
    ctx = flyte.ctx()
    if ctx:
        output_path = ctx.output_path + "/" + f"{card_type}.{format}"
        content_type = _FORMAT_CONTENT_TYPES.get(format, "application/octet-stream")
        attributes = {
            "Content-Type": content_type,  # For s3
            "content_type": content_type,  # For gcs
        }
        uri = await storage.put_stream(local_path.read_bytes(), to_path=output_path, attributes=attributes)
    else:
        import flyte.remote as remote

        _, uri = await remote.upload_file.aio(local_path)
    return Card(uri=uri, format=format, card_type=card_type)
