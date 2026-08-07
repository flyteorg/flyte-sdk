from __future__ import annotations

import hashlib
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
            try:
                return await _upload_card_from_local(temp_path, format=format, card_type=card_type)
            finally:
                # delete=False above means nothing else cleans this up.
                temp_path.unlink(missing_ok=True)
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
    content_type = _FORMAT_CONTENT_TYPES.get(format, "application/octet-stream")
    if ctx:
        data = local_path.read_bytes()
        # Content-address the object name. A fixed `{card_type}.{format}` name collides whenever a
        # task uploads more than one card of the same kind — two concurrent create_from calls would
        # silently overwrite each other, and the losing Card would hand back a URI holding the other
        # card's bytes. Hashing the content also makes re-uploading identical bytes idempotent.
        digest = hashlib.sha256(data).hexdigest()[:16]
        output_path = f"{ctx.output_path}/cards/{card_type}-{digest}.{format}"
        attributes = {
            "Content-Type": content_type,  # For s3
            "content_type": content_type,  # For gcs
        }
        uri = await storage.put_stream(data, to_path=output_path, attributes=attributes)
    else:
        import flyte.remote as remote

        # Same reason as the in-task branch above: without the MIME type the browser
        # downloads the card's presigned URL instead of rendering it.
        _, uri = await remote.upload_file.aio(local_path, content_type=content_type)
    return Card(uri=uri, format=format, card_type=card_type)
