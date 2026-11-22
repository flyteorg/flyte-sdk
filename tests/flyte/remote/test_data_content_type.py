"""Content-Type handling on the signed-URL PUT in _upload_single_file."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyte.remote._data import _is_gcs_v2_signed_url, _upload_single_file

_KEY = "org/proj/dev/FNKEDHLQPGLR6WH6AWKCDDO57Q======/script_generated.py"
GCS_V2_URL = (
    f"https://storage.googleapis.com/bucket/{_KEY}"
    "?Expires=1790201765&GoogleAccessId=sa%40proj.iam.gserviceaccount.com&Signature=abc%3D%3D"
)
GCS_V4_URL = (
    f"https://storage.googleapis.com/bucket/{_KEY}"
    "?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=sa%2F20260924%2Fauto%2Fstorage%2Fgoog4_request"
    "&X-Goog-Date=20260924T221500Z&X-Goog-Expires=3600"
    "&X-Goog-SignedHeaders=content-md5%3Bhost%3Bx-goog-meta-flytecontentmd5&X-Goog-Signature=abc"
)
S3_URL = (
    f"https://bucket.s3.us-east-2.amazonaws.com/{_KEY}"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA%2F20260924%2Fus-east-2%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260924T221500Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=abc"
)


@pytest.mark.parametrize(
    "url, expected",
    [
        (GCS_V2_URL, True),
        (GCS_V4_URL, False),
        (S3_URL, False),
        ("https://acct.blob.core.windows.net/c/k?sv=2021-08-06&sig=abc", False),
    ],
)
def test_is_gcs_v2_signed_url(url, expected):
    assert _is_gcs_v2_signed_url(url) is expected


def _make_cfg():
    cfg = MagicMock()
    cfg.project = "test-project"
    cfg.domain = "development"
    cfg.org = "test-org"
    return cfg


def _make_client(signed_url: str, headers: dict):
    resp = MagicMock()
    resp.signed_url = signed_url
    resp.native_url = "gs://bucket/" + _KEY if "googleapis" in signed_url else "s3://bucket/" + _KEY
    resp.headers = headers
    client = MagicMock()
    client.dataproxy_service.create_upload_location = AsyncMock(return_value=resp)
    return client


async def _put_headers(tmp_path, signed_url, server_headers, content_type):
    fp = tmp_path / "script_generated.py"
    fp.write_text("print(1)\n")
    upload = AsyncMock()
    with (
        patch("flyte._initialize._get_init_config", return_value=_make_cfg()),
        patch("flyte.remote._data.get_client", return_value=_make_client(signed_url, server_headers)),
        patch("flyte.remote._data._upload_with_retry", upload),
    ):
        await _upload_single_file(_make_cfg(), fp, content_type=content_type)
    upload.assert_awaited_once()
    return upload.await_args.kwargs["extra_headers"]


_SERVER_HEADERS = {"Content-MD5": "K1RBnXB5lx9Y/gWUIY3d/A==", "x-goog-meta-flyteContentMD5": "K1RBnXB5lx9Y/gWUIY3d/A=="}


@pytest.mark.asyncio
async def test_gcs_v2_signed_url_does_not_get_client_content_type(tmp_path):
    """GCS V2 signs Content-Type unconditionally; a header the signer never saw yields SignatureDoesNotMatch."""
    headers = await _put_headers(tmp_path, GCS_V2_URL, _SERVER_HEADERS, "text/x-python")
    assert not any(k.lower() == "content-type" for k in headers)
    # The signer-provided headers still go through untouched.
    assert headers["x-goog-meta-flyteContentMD5"] == "K1RBnXB5lx9Y/gWUIY3d/A=="


@pytest.mark.parametrize("url", [GCS_V4_URL, S3_URL])
@pytest.mark.asyncio
async def test_v4_style_signed_urls_keep_client_content_type(tmp_path, url):
    """GCS V4 and S3 SigV4 verify only the headers they list, so the MIME hint is safe to send."""
    headers = await _put_headers(tmp_path, url, _SERVER_HEADERS, "text/x-python")
    assert headers["Content-Type"] == "text/x-python"


@pytest.mark.asyncio
async def test_server_pinned_content_type_wins_even_on_gcs_v2(tmp_path):
    """A Content-Type the signing service returned is part of the signature and must be sent as-is."""
    headers = await _put_headers(tmp_path, GCS_V2_URL, {**_SERVER_HEADERS, "Content-Type": "text/html"}, "text/plain")
    assert headers["Content-Type"] == "text/html"
