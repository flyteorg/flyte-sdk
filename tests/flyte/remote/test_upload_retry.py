"""Tests for _upload_with_retry timeout and retry behavior."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from flyte.errors import RuntimeSystemError
from flyte.remote._data import (
    _UPLOAD_EXPIRES_IN,
    _UPLOAD_TIMEOUT,
    _is_expired_signed_url,
    _is_retryable_store_error,
    _redact_signed_url,
    _upload_with_retry,
)

# What S3 actually answers when the signature was valid but the clock ran out (FLYTE-SDK-5F).
S3_EXPIRED_BODY = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<Error><Code>AccessDenied</Code><Message>Request has expired</Message>"
    "<X-Amz-Expires>60</X-Amz-Expires><Expires>2026-08-01T01:18:50Z</Expires>"
    "<ServerTime>2026-08-01T01:18:50Z</ServerTime><RequestId>5ZA7RGJDKJDF8S3Z</RequestId></Error>"
)

# What S3 answers when the PUT body stopped arriving and it hung up on us (FLYTE-SDK-5F). The
# status is 400, which the retry table otherwise reads as "malformed, do not retry".
S3_REQUEST_TIMEOUT_BODY = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<Error><Code>RequestTimeout</Code><Message>Your socket connection to the server was not read "
    "from or written to within the timeout period. Idle connections will be closed.</Message>"
    "<RequestId>9PNQK5AEAFBKHM7E</RequestId></Error>"
)


@pytest.fixture
def upload_file(tmp_path):
    f = tmp_path / "bundle.tar.gz"
    f.write_bytes(b"fake bundle content")
    return f


@pytest.mark.asyncio
async def test_upload_success(upload_file):
    resp = httpx.Response(200)
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = resp
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        result = await _upload_with_retry(upload_file, "https://signed.url/upload", {}, verify=True)
        assert result.status_code == 200
        mock_cls.assert_called_with(verify=True, timeout=_UPLOAD_TIMEOUT)


@pytest.mark.asyncio
async def test_upload_timeout_default():
    assert _UPLOAD_TIMEOUT.read == 600.0
    assert _UPLOAD_TIMEOUT.connect == 30.0


@pytest.mark.asyncio
async def test_upload_timeout_env_override():
    with patch.dict("os.environ", {"FLYTE_UPLOAD_TIMEOUT": "120"}):
        import importlib

        import flyte.remote._data as data_mod

        importlib.reload(data_mod)
        assert data_mod._UPLOAD_TIMEOUT.read == 120.0
        assert data_mod._UPLOAD_TIMEOUT.connect == 30.0

        # Restore default
        del data_mod
        import flyte.remote._data

        importlib.reload(flyte.remote._data)


@pytest.mark.asyncio
async def test_upload_retries_on_timeout(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = httpx.ReadTimeout("timed out")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="timed out"):
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=2, min_backoff_sec=0.01
            )

        assert client.put.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_upload_retries_on_connect_error(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = httpx.ConnectError("connection refused")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="connection refused"):
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=1, min_backoff_sec=0.01
            )

        assert client.put.call_count == 2


@pytest.mark.asyncio
async def test_upload_retries_on_read_error(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = httpx.ReadError("connection reset by peer")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="connection reset by peer"):
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=1, min_backoff_sec=0.01
            )

        assert client.put.call_count == 2


@pytest.mark.asyncio
async def test_upload_retries_on_server_error_then_succeeds(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(503, text="service unavailable"),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        result = await _upload_with_retry(
            upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
        )
        assert result.status_code == 200
        assert client.put.call_count == 2


@pytest.mark.asyncio
async def test_upload_error_message_includes_type_when_empty(upload_file):
    # httpx.ReadError() with no message used to surface as
    # "Failed to upload ... after N retries: " (trailing colon, no cause).
    # The error message must include the exception type so the failure stays actionable.
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = httpx.ReadError("")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="ReadError") as exc_info:
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=1, min_backoff_sec=0.01
            )

        # Must not end with a bare ": " (the empty-cause regression).
        assert not str(exc_info.value).rstrip().endswith("retries:")


@pytest.mark.asyncio
async def test_upload_no_retry_on_client_error(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = httpx.Response(403, text="forbidden")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="status 403"):
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
            )

        assert client.put.call_count == 1  # no retries for 4xx


@pytest.mark.asyncio
async def test_upload_honors_retry_after_seconds(upload_file):
    """When the server returns 429 with Retry-After: <int>, we sleep that long."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(429, headers={"Retry-After": "2"}, text="slow down"),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with patch("flyte.remote._data.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
            )

        assert result.status_code == 200
        assert client.put.call_count == 2
        # Honored the Retry-After value (2s) rather than the exponential value (~0.01s).
        mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_upload_caps_absurd_retry_after(upload_file):
    """A misbehaving server returning Retry-After: 99999 should be clamped."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(429, headers={"Retry-After": "99999"}, text="slow down"),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with patch("flyte.remote._data.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await _upload_with_retry(
                upload_file,
                "https://signed.url/upload",
                {},
                verify=True,
                max_retries=3,
                min_backoff_sec=0.01,
                retry_after_cap_sec=5.0,
            )

        assert result.status_code == 200
        mock_sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_upload_429_without_retry_after_uses_exponential(upload_file):
    """If no Retry-After header is sent, normal exponential backoff applies."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(429, text="slow down"),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with patch("flyte.remote._data.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await _upload_with_retry(
                upload_file,
                "https://signed.url/upload",
                {},
                verify=True,
                max_retries=3,
                min_backoff_sec=0.01,
                max_backoff_sec=10.0,
            )

        assert result.status_code == 200
        # Exponential value for the first retry: min_backoff_sec * 2**0 = 0.01
        mock_sleep.assert_awaited_once_with(0.01)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://bucket.s3.amazonaws.com/org/key.tar.gz", "https://bucket.s3.amazonaws.com/org/key.tar.gz"),
        (
            "https://bucket.s3.amazonaws.com/org/key.tar.gz?X-Amz-Signature=deadbeef&X-Amz-Security-Token=secret",
            "https://bucket.s3.amazonaws.com/org/key.tar.gz?<redacted>",
        ),
        ("https://bucket.s3.amazonaws.com/key?", "https://bucket.s3.amazonaws.com/key?<redacted>"),
    ],
)
def test_redact_signed_url(url, expected):
    assert _redact_signed_url(url) == expected


@pytest.mark.asyncio
async def test_upload_client_error_message_redacts_signed_url(upload_file):
    """FLYTE-SDK-6R: the non-retryable-error message must not leak signed-URL credentials."""
    signed_url = (
        "https://bucket.s3.us-west-2.amazonaws.com/org/proj/bundle.tar.gz"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAEXAMPLE"
        "&X-Amz-Security-Token=SUPERSECRETTOKEN&X-Amz-Signature=deadbeef"
    )
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = httpx.Response(403, text="forbidden")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError) as exc_info:
            await _upload_with_retry(upload_file, signed_url, {}, verify=True, max_retries=0)

    message = str(exc_info.value)
    assert "SUPERSECRETTOKEN" not in message
    assert "X-Amz-Signature" not in message
    # The bucket/key is still there, so the message stays diagnostic.
    assert "https://bucket.s3.us-west-2.amazonaws.com/org/proj/bundle.tar.gz?<redacted>" in message


def test_signed_url_outlives_a_full_upload_attempt():
    """FLYTE-SDK-5F: the URL must not expire while an upload we're still waiting on is in flight.

    The SDK used to ask for a 60s URL while allowing a single PUT to run for 600s, so any bundle
    that took longer than a minute to push died on an opaque S3 "403 Request has expired".
    """
    assert _UPLOAD_EXPIRES_IN.total_seconds() > _UPLOAD_TIMEOUT.read


def test_expires_in_tracks_a_raised_upload_timeout():
    """Raising FLYTE_UPLOAD_TIMEOUT must carry the URL lifetime with it, or the two disagree again."""
    import importlib

    with patch.dict("os.environ", {"FLYTE_UPLOAD_TIMEOUT": "1800"}):
        import flyte.remote._data as data_mod

        importlib.reload(data_mod)
        assert data_mod._UPLOAD_EXPIRES_IN.total_seconds() > data_mod._UPLOAD_TIMEOUT.read

    import flyte.remote._data

    importlib.reload(flyte.remote._data)


def test_derived_expires_in_stays_under_the_platform_cap():
    """A huge FLYTE_UPLOAD_TIMEOUT must not derive an expires_in the control plane will reject.

    dataproxy validates expires_in against ``upload.maxExpiresIn`` (1h by default) and fails the
    CreateUploadLocation outright, so an over-derived default would break uploads entirely rather
    than merely shortening them.
    """
    import importlib

    with patch.dict("os.environ", {"FLYTE_UPLOAD_TIMEOUT": "7200"}):
        import flyte.remote._data as data_mod

        importlib.reload(data_mod)
        assert data_mod._UPLOAD_EXPIRES_IN.total_seconds() == 3600.0

    import flyte.remote._data

    importlib.reload(flyte.remote._data)


def test_expires_in_env_override():
    import importlib

    with patch.dict("os.environ", {"FLYTE_UPLOAD_EXPIRES_IN": "3000"}):
        import flyte.remote._data as data_mod

        importlib.reload(data_mod)
        assert data_mod._UPLOAD_EXPIRES_IN.total_seconds() == 3000.0

    # An explicit value above the default platform cap is honored, not clamped: it is the escape
    # hatch for a deployment that raised its own upload.maxExpiresIn.
    with patch.dict("os.environ", {"FLYTE_UPLOAD_EXPIRES_IN": "7200"}):
        import flyte.remote._data as data_mod

        importlib.reload(data_mod)
        assert data_mod._UPLOAD_EXPIRES_IN.total_seconds() == 7200.0

    import flyte.remote._data

    importlib.reload(flyte.remote._data)


@pytest.mark.parametrize(
    "status_code, body, expected",
    [
        (403, S3_EXPIRED_BODY, True),
        (403, "<Error><Code>ExpiredToken</Code><Message>The provided token has expired.</Message></Error>", True),
        (400, "<Error><Code>ExpiredToken</Code></Error>", True),
        (
            403,
            "<?xml version='1.0'?><Error><Code>AuthenticationFailed</Code>"
            "<Message>Signature not valid in the specified time frame</Message></Error>",
            True,
        ),
        # A genuine authorization failure is not an expiry, and needs different advice.
        (403, "<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>", False),
        (403, "forbidden", False),
        # Only authorization statuses carry the expiry meaning; don't reinterpret a 500.
        (500, S3_EXPIRED_BODY, False),
    ],
)
def test_is_expired_signed_url(status_code, body, expected):
    assert _is_expired_signed_url(status_code, body) is expected


@pytest.mark.asyncio
async def test_expired_url_gets_its_own_actionable_error(upload_file):
    """An expired URL must say so — and point at the knob — instead of dumping S3's XML."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = httpx.Response(403, text=S3_EXPIRED_BODY)
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError) as exc_info:
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
            )

    message = str(exc_info.value)
    assert "expired" in message
    assert "FLYTE_UPLOAD_EXPIRES_IN" in message
    # Re-PUTting the same expired signature can only fail the same way.
    assert client.put.call_count == 1


@pytest.mark.asyncio
async def test_expired_url_error_redacts_credentials(upload_file):
    """The expiry branch must redact like every other message (FLYTE-SDK-6R)."""
    signed_url = (
        "https://bucket.s3.us-west-2.amazonaws.com/org/proj/bundle.tar.gz"
        "?X-Amz-Credential=ASIAEXAMPLE&X-Amz-Security-Token=SUPERSECRETTOKEN&X-Amz-Signature=deadbeef"
    )
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = httpx.Response(403, text=S3_EXPIRED_BODY)
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError) as exc_info:
            await _upload_with_retry(upload_file, signed_url, {}, verify=True, max_retries=0)

    message = str(exc_info.value)
    assert "SUPERSECRETTOKEN" not in message
    assert "https://bucket.s3.us-west-2.amazonaws.com/org/proj/bundle.tar.gz?<redacted>" in message


@pytest.mark.parametrize(
    "status_code, body, expected",
    [
        (400, S3_REQUEST_TIMEOUT_BODY, True),
        # The code match is on the store's own <Code> element, not loose prose.
        (400, "<Error><Code>requesttimeout</Code></Error>", True),
        (400, "<Error><Code>InvalidArgument</Code><Message>bad request</Message></Error>", False),
        (400, "<Error><Code>ExpiredToken</Code></Error>", False),
        # A message that merely mentions a timeout is not the store asking for a retry.
        (400, "the request timed out", False),
        # Only 400 is reinterpreted; every other status already means what the table says.
        (403, S3_REQUEST_TIMEOUT_BODY, False),
        (500, S3_REQUEST_TIMEOUT_BODY, False),
    ],
)
def test_is_retryable_store_error(status_code, body, expected):
    assert _is_retryable_store_error(status_code, body) is expected


@pytest.mark.asyncio
async def test_stalled_upload_is_retried_instead_of_failing(upload_file):
    """S3's `400 RequestTimeout` means "the socket went idle, send it again", not "give up"."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(400, text=S3_REQUEST_TIMEOUT_BODY),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        result = await _upload_with_retry(
            upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
        )

    assert result.status_code == 200
    assert client.put.call_count == 2


@pytest.mark.asyncio
async def test_stalled_upload_that_never_recovers_reports_the_retries(upload_file):
    """A stall that outlasts the budget still fails — but as a retried upload, not a hard 400."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = httpx.Response(400, text=S3_REQUEST_TIMEOUT_BODY)
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="after 2 retries") as exc_info:
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=2, min_backoff_sec=0.01
            )

    assert "RequestTimeout" in str(exc_info.value)
    assert client.put.call_count == 3  # initial attempt + 2 retries


@pytest.mark.asyncio
async def test_expired_token_still_beats_the_retry_branch(upload_file):
    """`400 ExpiredToken` shares the status but not the meaning — it must stay unretried."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = httpx.Response(400, text="<Error><Code>ExpiredToken</Code></Error>")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError) as exc_info:
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
            )

    assert "expired" in str(exc_info.value)
    assert client.put.call_count == 1
