from __future__ import annotations

import hashlib
import json
import typing
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode

from flyte.io import PARQUET, DataFrame

_HF_PARQUET_REVISION = "refs/convert/parquet"


@dataclass
class HFSource:
    """HuggingFace dataset source for task parameter defaults."""

    repo: str
    name: str | None = None
    split: str | None = None
    revision: str | None = None
    cache_root: str | None = None

    def to_hf_uri(self) -> str:
        uri = f"hf://{self.repo}"
        params = {}
        if self.name:
            params["name"] = self.name
        if self.split:
            params["split"] = self.split
        if self.cache_root:
            params["cache_root"] = self.cache_root
        if self.revision:
            params["revision"] = self.revision
        if params:
            uri = f"{uri}?{urlencode(params, safe=':/')}"
        return uri

    @classmethod
    def from_hf_uri(cls, uri: str) -> "HFSource":
        if not uri.startswith("hf://"):
            raise ValueError(f"Invalid HF URI: {uri}")

        path, _, query = uri[5:].partition("?")
        repo = path.strip("/")
        if not repo:
            raise ValueError(f"Invalid HF URI: {uri}")

        query_params = parse_qs(query)
        name = query_params.get("name", [None])[0]
        split = query_params.get("split", [None])[0]
        revision = query_params.get("revision", [None])[0]
        cache_root = query_params.get("cache_root", [None])[0]

        return cls(
            repo=repo,
            name=name,
            split=split,
            revision=revision,
            cache_root=cache_root,
        )


def from_hf(
    repo: str,
    *,
    name: str | None = None,
    split: str | None = None,
    revision: str | None = None,
    cache_root: str | None = None,
) -> DataFrame:
    """Return a DataFrame reference for use as a task parameter default.

    cache_root optionally points at a stable remote directory that can be reused
    across runs. Without it, the dataset is materialized to a generated Flyte raw-data
    path for this run. If name is omitted, the plugin resolves the dataset's
    default converted-parquet config, or the only available config when there is
    exactly one.
    """
    source = HFSource(
        repo=repo,
        name=name,
        split=split,
        revision=revision,
        cache_root=cache_root,
    )

    return DataFrame(
        uri=source.to_hf_uri(),
        format=PARQUET,
        hash=hf_source_cache_key(source),
    )


@dataclass(frozen=True)
class HFShard:
    rel_path: str
    hf_name: str
    size: int | None = None
    etag: str | None = None
    last_modified: str | None = None


def hf_revision(source: HFSource) -> str:
    return source.revision or _HF_PARQUET_REVISION


def hf_source_payload(source: HFSource, shards: list[HFShard] | None = None) -> dict[str, typing.Any]:
    payload: dict[str, typing.Any] = {
        "repo": source.repo,
        "name": source.name,
        "split": source.split,
        "revision": hf_revision(source),
    }
    if shards is not None:
        payload["shards"] = [
            {
                "rel_path": shard.rel_path,
                "hf_name": shard.hf_name,
                "size": shard.size,
                "etag": shard.etag,
                "last_modified": (str(shard.last_modified) if shard.last_modified is not None else None),
            }
            for shard in sorted(shards, key=lambda s: s.rel_path)
        ]
    return payload


def hf_source_cache_key(source: HFSource, shards: list[HFShard] | None = None) -> str:
    payload = json.dumps(
        hf_source_payload(source, shards),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
