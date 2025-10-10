import asyncio
import hashlib
import os
import typing
import uuid
from base64 import b64encode
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import aiofiles
import grpc
import httpx
from flyteidl.service import dataproxy_pb2
from google.protobuf import duration_pb2

from flyte._initialize import CommonInit, ensure_client, get_client, get_init_config, require_project_and_domain
from flyte.errors import InitializationError, RuntimeSystemError
from flyte.syncify import syncify

_UPLOAD_EXPIRES_IN = timedelta(seconds=60)


def get_extra_headers_for_protocol(native_url: str) -> typing.Dict[str, str]:
    """
    For Azure Blob Storage, we need to set certain headers for http request.
    This is used when we work with signed urls.
    :param native_url:
    :return:
    """
    if native_url.startswith("abfs://"):
        return {"x-ms-blob-type": "BlockBlob"}
    return {}


@lru_cache
def hash_file(file_path: typing.Union[os.PathLike, str]) -> Tuple[bytes, str, int]:
    """
    Hash a file and produce a digest to be used as a version
    """
    h = hashlib.md5()
    size = 0

    with open(file_path, "rb") as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)

    return h.digest(), h.hexdigest(), size


@require_project_and_domain
async def _upload_single_file(
    cfg: CommonInit, fp: Path, verify: bool = True, basedir: str | None = None
) -> Tuple[str, str]:
    md5_bytes, str_digest, _ = hash_file(fp)
    from flyte._logging import logger

    try:
        expires_in_pb = duration_pb2.Duration()
        expires_in_pb.FromTimedelta(_UPLOAD_EXPIRES_IN)
        client = get_client()
        resp = await client.dataproxy_service.CreateUploadLocation(  # type: ignore
            dataproxy_pb2.CreateUploadLocationRequest(
                project=cfg.project,
                domain=cfg.domain,
                content_md5=md5_bytes,
                filename=fp.name,
                expires_in=expires_in_pb,
                filename_root=basedir,
                add_content_md5_metadata=True,
            )
        )
    except grpc.aio.AioRpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise RuntimeSystemError(
                "NotFound", f"Failed to get signed url for {fp}, please check your project and domain: {e.details()}"
            )
        elif e.code() == grpc.StatusCode.PERMISSION_DENIED:
            raise RuntimeSystemError(
                "PermissionDenied", f"Failed to get signed url for {fp}, please check your permissions: {e.details()}"
            )
        elif e.code() == grpc.StatusCode.UNAVAILABLE:
            raise InitializationError("EndpointUnavailable", "user", "Service is unavailable.")
        else:
            raise RuntimeSystemError(e.code().value, f"Failed to get signed url for {fp}: {e.details()}")
    except Exception as e:
        raise RuntimeSystemError(type(e).__name__, f"Failed to get signed url for {fp}.") from e
    logger.debug(f"Uploading to [link={resp.signed_url}]signed url[/link] for [link=file://{fp}]{fp}[/link]")
    extra_headers = get_extra_headers_for_protocol(resp.native_url)
    extra_headers.update(resp.headers)
    encoded_md5 = b64encode(md5_bytes)
    content_length = fp.stat().st_size

    async with aiofiles.open(str(fp), "rb") as file:
        extra_headers.update({"Content-Length": str(content_length), "Content-MD5": encoded_md5.decode("utf-8")})
        async with httpx.AsyncClient(verify=verify) as aclient:
            put_resp = await aclient.put(resp.signed_url, headers=extra_headers, content=file)
            if put_resp.status_code not in [200, 201, 204]:
                raise RuntimeSystemError(
                    "UploadFailed",
                    f"Failed to upload {fp} to {resp.signed_url}, status code: {put_resp.status_code}, "
                    f"response: {put_resp.text}",
                )
        # TODO in old code we did this
        #             if self._config.platform.insecure_skip_verify is True
        #             else self._config.platform.ca_cert_file_path,
    logger.debug(f"Uploaded with digest {str_digest}, blob location is {resp.native_url}")
    return str_digest, resp.native_url


@syncify
async def upload_file(fp: Path, verify: bool = True) -> Tuple[str, str]:
    """
    Uploads a file to a remote location and returns the remote URI.

    :param fp: The file path to upload.
    :param verify: Whether to verify the certificate for HTTPS requests.
    :return: A tuple containing the MD5 digest and the remote URI.
    """
    # This is a placeholder implementation. Replace with actual upload logic.
    ensure_client()
    cfg = get_init_config()
    if not fp.is_file():
        raise ValueError(f"{fp} is not a single file, upload arg must be a single file.")
    return await _upload_single_file(cfg, fp, verify=verify)


async def upload_dir(dir_path: Path, verify: bool = True) -> str:
    """
    Uploads a directory to a remote location and returns the remote URI.

    :param dir_path: The directory path to upload.
    :param verify: Whether to verify the certificate for HTTPS requests.
    :return: The remote URI of the uploaded directory.
    """
    # This is a placeholder implementation. Replace with actual upload logic.
    ensure_client()
    cfg = get_init_config()
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory, upload arg must be a directory.")

    prefix = uuid.uuid4().hex

    files = dir_path.rglob("*")
    uploaded_files = []
    for file in files:
        if file.is_file():
            uploaded_files.append(_upload_single_file(cfg, file, verify=verify, basedir=prefix))

    urls = await asyncio.gather(*uploaded_files)
    native_url = urls[0][1]  # Assuming all files are uploaded to the same prefix
    # native_url is of the form s3://my-s3-bucket/flytesnacks/development/{prefix}/source/empty.md
    uri = native_url.split(prefix)[0] + "/" + prefix

    return uri
