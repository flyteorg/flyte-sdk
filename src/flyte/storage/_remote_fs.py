from __future__ import annotations

import os
import pathlib
import tempfile
import threading
import typing

import fsspec
import fsspec.callbacks
import fsspec.utils
from fsspec.asyn import AsyncFileSystem

from flyte._logging import logger

# This file system is not really a filesystem, so users aren't really able to specify the remote path,
# at least not yet.
REMOTE_PLACEHOLDER = "flyte://data"
_PREFIX_KEY = "upload_prefix"

HashStructure = typing.Dict[str, typing.Tuple[bytes, int]]


class RemoteFSPathResolver:
    protocol = "flyte://"
    _flyte_path_to_remote_map: typing.ClassVar[typing.Dict[str, str]] = {}
    _lock = threading.Lock()

    @classmethod
    def resolve_remote_path(cls, flyte_uri: str) -> typing.Optional[str]:
        """
        Given a flyte uri, return the remote path if it exists or was created in current session, otherwise return None
        """
        with cls._lock:
            if flyte_uri in cls._flyte_path_to_remote_map:
                resolved = cls._flyte_path_to_remote_map[flyte_uri]
                logger.debug(f"Resolved {flyte_uri} -> {resolved}")
                return resolved
            logger.warning(
                f"Failed to resolve {flyte_uri}. Available mappings: {list(cls._flyte_path_to_remote_map.keys())}"
            )
            return None

    @classmethod
    def add_mapping(cls, flyte_uri: str, remote_path: str):
        """
        Thread safe method to dd a mapping from a flyte uri to a remote path
        """
        with cls._lock:
            cls._flyte_path_to_remote_map[flyte_uri] = remote_path


class HttpFileWriter(fsspec.asyn.AbstractAsyncStreamedFile):
    def __init__(self, filename: str, **kwargs):
        self._filename = filename
        # Create a temporary file to buffer chunks before upload
        # Sanitize filename for use in suffix
        safe_suffix = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)[:50]
        self._tmp_file_obj = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=f"_{safe_suffix}")
        self._tmp_file_path = pathlib.Path(self._tmp_file_obj.name)
        super().__init__(**kwargs)

    async def _upload_chunk(self, final=False):
        """Only uploads the file at once from the buffer.
        Not suitable for large files as the buffer will blow the memory for very large files.
        Suitable for default values or local dataframes being uploaded all at once.

        This function is called by fsspec.flush(). This will create a new file upload location.
        """
        import flyte.remote as remote

        if final is False:
            # Write buffer contents to temp file
            buffer_data = self.buffer.getvalue()
            if buffer_data:
                self._tmp_file_obj.write(buffer_data)
                self._tmp_file_obj.flush()
            self.buffer.seek(0)
            self.buffer.truncate(0)
            return False

        # Final upload: write any remaining buffer content, close temp file, and upload
        buffer_data = self.buffer.getvalue()
        if buffer_data:
            self._tmp_file_obj.write(buffer_data)
        self._tmp_file_obj.flush()
        self._tmp_file_obj.close()

        try:
            # Upload the temp file synchronously using remote.upload_file
            _, native_uri = await remote.upload_file.aio(fp=self._tmp_file_path)
            # Add mapping with the flyte:// protocol prefix
            flyte_uri = f"flyte://{self.path}"
            RemoteFSPathResolver.add_mapping(flyte_uri, native_uri)
            return True
        finally:
            # Clean up the temporary file
            if self._tmp_file_path.exists():
                self._tmp_file_path.unlink()


class FlyteFS(AsyncFileSystem):
    """
    Want this to behave mostly just like the HTTP file system.
    """

    sep = "/"
    protocol = "flyte"

    def __init__(
        self,
        asynchronous: bool = False,
        **storage_options,
    ):
        super().__init__(asynchronous=asynchronous, **storage_options)

    @property
    def fsid(self) -> str:
        return "flyte"

    async def _get_file(self, rpath, lpath, **kwargs):
        """
        Don't do anything special. If it's a flyte url, the create a download link and write to lpath,
        otherwise default to parent.
        """
        raise NotImplementedError("FlyteFS currently doesn't support downloading files.")

    async def _put_file(
        self,
        lpath,
        rpath,
        chunk_size=5 * 2**20,
        callback=fsspec.callbacks.DEFAULT_CALLBACK,
        method="put",
        **kwargs,
    ):
        """
        fsspec will call this method to upload a file. If recursive, rpath will already be individual files.
        Make the request and upload, but then how do we get the s3 paths back to the user?
        """
        import flyte.remote as remote

        prefix = None
        if _PREFIX_KEY in kwargs:
            prefix = kwargs[_PREFIX_KEY]

        native_uri = await remote.upload_dir(pathlib.Path(lpath), prefix=prefix)
        return native_uri

    @staticmethod
    def extract_common(native_urls: typing.List[str]) -> str:
        """
        This function that will take a list of strings and return the longest prefix that they all have in common.
        That is, if you have
            ['s3://my-s3-bucket/flytesnacks/development/ABCYZWMPACZAJ2MABGMOZ6CCPY======/source/empty.md',
             's3://my-s3-bucket/flytesnacks/development/ABCXKL5ZZWXY3PDLM3OONUHHME======/source/nested/more.txt',
             's3://my-s3-bucket/flytesnacks/development/ABCXBAPBKONMADXVW5Q3J6YBWM======/source/original.txt']
        this will return back 's3://my-s3-bucket/flytesnacks/development/'
        Note that trailing characters after a separator that just happen to be the same will also be stripped.
        """
        if len(native_urls) == 0:
            return ""
        if len(native_urls) == 1:
            return native_urls[0]

        common_prefix = ""
        shortest = min([len(x) for x in native_urls])
        x = [[native_urls[j][i] for j in range(len(native_urls))] for i in range(shortest)]
        for i in x:
            if len(set(i)) == 1:
                common_prefix += i[0]
            else:
                break

        fs_class = fsspec.get_filesystem_class(fsspec.utils.get_protocol(native_urls[0]))
        sep = fs_class.sep
        # split the common prefix on the last separator so we don't get any trailing characters.
        common_prefix = common_prefix.rsplit(sep, 1)[0]
        logger.debug(f"Returning {common_prefix} from {native_urls}")
        return common_prefix

    @staticmethod
    def get_filename_root(file_info: HashStructure) -> str:
        """
        Given a dictionary of file paths to hashes and content lengths, return a consistent filename root.
        This is done by hashing the sorted list of file paths and then base32 encoding the result.
        If the input is empty, then generate a random string
        """
        import base64
        import hashlib
        import random
        import uuid

        if len(file_info) == 0:
            return uuid.UUID(int=random.getrandbits(128)).hex
        sorted_paths = sorted(file_info.keys())
        h = hashlib.md5()
        for p in sorted_paths:
            h.update(file_info[p][0])
        return base64.b32encode(h.digest()).decode("utf-8")

    def get_hashes_and_lengths(self, p: pathlib.Path) -> HashStructure:
        """
        Returns a flat list of absolute file paths to their hashes and content lengths
        this output is used both for the file upload request, and to create consistently a filename root for
        uploaded folders. We'll also use it for single files just for consistency.
        If a directory then all the files in the directory will be hashed.
        If a single file then just that file will be hashed.
        Skip symlinks
        """
        if p.is_symlink():
            return {}
        if p.is_dir():
            hashes = {}
            for f in p.iterdir():
                hashes.update(self.get_hashes_and_lengths(f))
            return hashes
        else:
            from flyte.remote._data import hash_file

            md5_bytes, _, content_length = hash_file(p.resolve())
            return {str(p.absolute()): (md5_bytes, content_length)}

    async def _put(
        self,
        lpath,
        rpath,
        recursive=False,
        callback=fsspec.callbacks.DEFAULT_CALLBACK,
        batch_size=None,
        **kwargs,
    ):
        """
        cp file.txt flyte://data/...
        rpath gets ignored, so it doesn't matter what it is.
        """
        # Hash everything at the top level
        # TODO support prefix, as the hash of various files etc
        file_info = self.get_hashes_and_lengths(pathlib.Path(lpath))
        prefix = self.get_filename_root(file_info)

        kwargs[_PREFIX_KEY] = prefix
        res = await super()._put(lpath, REMOTE_PLACEHOLDER, recursive, callback, batch_size, **kwargs)
        if isinstance(res, list):
            res = self.extract_common(res)
        RemoteFSPathResolver.add_mapping(rpath.strip(os.path.sep), res)
        return res

    async def _isdir(self, path):
        return True

    def exists(self, path, **kwargs):
        raise NotImplementedError("flyte file system currently can't check if a file exists.")

    def _open(
        self,
        path,
        mode="wb",
        block_size=None,
        autocommit=None,  # XXX: This differs from the base class.
        cache_type=None,
        cache_options=None,
        size=None,
        **kwargs,
    ):
        if mode != "wb":
            raise ValueError("Only wb mode is supported")

        # Dataframes are written as multiple files, default is the first file with 00000 suffix, we should drop
        # that suffix and use the parent directory as the remote path.

        return HttpFileWriter(os.path.basename(path), fs=self, path=os.path.dirname(path), mode=mode, **kwargs)

    def __str__(self):
        p = super().__str__()
        return f"FlyteFS(): {p}"
