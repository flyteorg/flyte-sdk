"""
The image lookup table handed from the process that builds images to the tasks it launches.

Lives apart from `flyte._internal.imagebuild`, which carries the builders, the image checkers and
the local persistence layer. A task container only needs to decode the cache it was handed, so the
runtime entrypoint imports this module and nothing from the build machinery.
"""

from __future__ import annotations

import typing
from typing import Dict, Tuple

from pydantic import BaseModel


class RunIdentifierData(BaseModel):
    org: str
    project: str
    domain: str
    name: str


class ImageCache(BaseModel):
    image_lookup: Dict[str, str]
    build_run_ids: Dict[str, RunIdentifierData] = {}
    serialized_form: str | None = None

    @property
    def to_transport(self) -> str:
        """
        Returns:
            returns the serialization context as a base64encoded, gzip compressed, json string
        """
        # This is so that downstream tasks continue to have the same image lookup abilities
        import base64
        import gzip
        from io import BytesIO

        if self.serialized_form:
            return self.serialized_form
        json_str = self.model_dump_json(exclude={"serialized_form"})
        buf = BytesIO()
        with gzip.GzipFile(mode="wb", fileobj=buf, mtime=0) as f:
            f.write(json_str.encode("utf-8"))
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @classmethod
    def from_transport(cls, s: str) -> ImageCache:
        import base64
        import gzip

        compressed_val = base64.b64decode(s.encode("utf-8"))
        json_str = gzip.decompress(compressed_val).decode("utf-8")
        val = cls.model_validate_json(json_str)
        val.serialized_form = s
        return val

    def repr(self) -> typing.List[typing.List[Tuple[str, str]]]:
        """
        Returns a detailed representation of the deployed environments.
        """
        tuples = []
        for k, v in self.image_lookup.items():
            tuples.append(
                [
                    ("Name", k),
                    ("image", v),
                ]
            )
        return tuples
