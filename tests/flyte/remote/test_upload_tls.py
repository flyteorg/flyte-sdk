"""The trust configured for the session has to reach object-store uploads too.

`verify=True` used to mean "certifi", which left the control plane and the object
store on two different trust stores: `admin.caCertFilePath` fixed the first and was
invisible to the second, so uploads failed with CERTIFICATE_VERIFY_FAILED on a
connection the control plane had just made successfully.
"""

import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import certifi
import httpx
import pytest

from flyte.errors import InitializationError
from flyte.remote._data import (
    _UPLOAD_TIMEOUT,
    _resolve_upload_verify,
    _ssl_context_from_ca_file,
    _upload_with_retry,
)


def _session(*, ca_cert_file_path=None, insecure_skip_verify=False):
    auth_kwargs = {"ca_cert_file_path": ca_cert_file_path} if ca_cert_file_path else {}
    return MagicMock(auth_kwargs=auth_kwargs, insecure_skip_verify=insecure_skip_verify)


def _patch_session(session):
    return patch("flyte.remote._data.get_client", return_value=MagicMock(session_config=session))


def test_ca_cert_file_path_reaches_uploads():
    with _patch_session(_session(ca_cert_file_path=certifi.where())):
        resolved = _resolve_upload_verify(True)
    assert isinstance(resolved, ssl.SSLContext)
    # Trusting exactly that bundle, not silently falling back to something else.
    assert len(resolved.get_ca_certs()) == len(ssl.create_default_context(cafile=certifi.where()).get_ca_certs())


def test_ca_context_is_cached_per_path():
    """upload_dir fans out per file; the bundle must be parsed once, not per PUT."""
    assert _ssl_context_from_ca_file(certifi.where()) is _ssl_context_from_ca_file(certifi.where())


def test_insecure_skip_verify_does_not_verify():
    """The bootstrapped chain anchors the control plane's host, not the object store's."""
    with _patch_session(_session(insecure_skip_verify=True)):
        assert _resolve_upload_verify(True) is False


def test_ca_cert_file_path_wins_over_skip_verify():
    """Matches _platform_to_client_kwargs, where ca_cert_file_path takes precedence."""
    with _patch_session(_session(ca_cert_file_path=certifi.where(), insecure_skip_verify=True)):
        assert isinstance(_resolve_upload_verify(True), ssl.SSLContext)


def test_plain_session_keeps_httpx_default():
    with _patch_session(_session()):
        assert _resolve_upload_verify(True) is True


@pytest.mark.parametrize("explicit", [False, ssl.create_default_context()])
def test_explicit_verify_is_untouched(explicit):
    """A caller who passed their own value should not have it overridden."""
    with _patch_session(_session(ca_cert_file_path=certifi.where())) as get_client:
        assert _resolve_upload_verify(explicit) is explicit
        get_client.assert_not_called()


def test_uninitialized_client_falls_back_to_default():
    with patch(
        "flyte.remote._data.get_client",
        side_effect=InitializationError("ClientNotInitializedError", "user", "nope"),
    ):
        assert _resolve_upload_verify(True) is True


@pytest.mark.asyncio
async def test_upload_passes_resolved_context_to_httpx(tmp_path):
    """End of the wire: the context reaches the client that makes the PUT."""
    f = tmp_path / "bundle.tar.gz"
    f.write_bytes(b"fake bundle content")

    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = httpx.Response(200)
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with _patch_session(_session(ca_cert_file_path=certifi.where())):
            await _upload_with_retry(f, "https://signed.url/upload", {}, verify=True)

    kwargs = mock_cls.call_args.kwargs
    assert isinstance(kwargs["verify"], ssl.SSLContext)
    assert kwargs["timeout"] == _UPLOAD_TIMEOUT
