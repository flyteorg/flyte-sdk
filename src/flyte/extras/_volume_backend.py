from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass(frozen=True)
class _MountConfig:
    meta_dir: str
    redis_port: Optional[int]
    cache_dir: str
    writeback: bool
    upload_delay: Optional[str]
    max_uploads: int
    attr_cache: float
    entry_cache: float
    dir_entry_cache: float
    # When True, the mount is read-only — writes through the FUSE mount
    # return EROFS. The metadata engine is still started (Redis loads the
    # index) but no chunk uploads can happen. Useful for "expose a prior
    # volume's state without forking" workflows.
    read_only: bool = False


@dataclass(frozen=True)
class _MetadataEngineState:
    proc: subprocess.Popen
    port: int


class _VolumeBackend(Protocol):
    name: str
    client_binary: str

    def index_filename(self, engine: str) -> str: ...

    def meta_url(self, meta_dir: str, engine: str, *, redis_port: Optional[int] = None) -> str: ...

    async def start_metadata_engine(self, meta_dir: str, engine: str) -> Optional[_MetadataEngineState]: ...

    def stop_metadata_engine(self, state: Optional[_MetadataEngineState], timeout: float) -> None: ...

    def format(self, *, storage: str, bucket: str, meta_url: str, name: str) -> None: ...

    def dump_metadata(self, meta_url: str, dump_path: str) -> None: ...

    def load_metadata(self, meta_url: str, dump_path: str) -> None: ...

    def mount_cmd(self, *, config: _MountConfig, engine: str, mount_path: str) -> list[str]: ...

    def is_mounted(self, mount_path: str) -> bool: ...

    def unmount(self, mount_path: str, *, flush: bool = False) -> None: ...

    def sync_filesystem(self, path: str) -> None: ...

    def save_metadata(self, engine: str, *, redis_port: Optional[int] = None) -> None: ...

    def checkpoint_metadata(self, index_path: str) -> None: ...

    def snapshot_index(self, src: Path, engine: str, tmp_prefix: str) -> str: ...

    async def query_stats(
        self, meta_dir: str, engine: str, *, redis_port: Optional[int] = None
    ) -> tuple[Optional[int], Optional[int]]: ...

    def disjoint_fork_counters(self, index_path: str, engine: str, offset: int) -> int: ...
