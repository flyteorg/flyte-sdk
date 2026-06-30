import errno
from unittest import mock

import click
import pytest

from flyte import _sentry


@pytest.mark.parametrize(
    "exc",
    [
        click.Abort(),
        click.exceptions.Exit(1),
        click.ClickException("docker daemon not running"),
    ],
)
def test_capture_exception_skips_user_errors(exc):
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(exc)
    init_mock.assert_not_called()


def test_capture_exception_reports_real_errors():
    with (
        mock.patch.object(_sentry, "init"),
        mock.patch("sentry_sdk.is_initialized", return_value=True),
        mock.patch("sentry_sdk.capture_exception") as capture_mock,
        mock.patch("sentry_sdk.flush"),
    ):
        err = RuntimeError("boom")
        _sentry.capture_exception(err)
    capture_mock.assert_called_once_with(err)


def test_capture_errors_decorator_filters_click_abort():
    @_sentry.capture_errors
    def fn():
        raise click.Abort

    with mock.patch.object(_sentry, "init") as init_mock:
        with pytest.raises(click.Abort):
            fn()
    init_mock.assert_not_called()


def test_capture_exception_skips_deployment_error():
    from flyte.errors import DeploymentError

    err = DeploymentError("bad trigger config")
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_image_build_error():
    from flyte.errors import ImageBuildError

    err = ImageBuildError("build failed")
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_initialization_error():
    from flyte.errors import InitializationError

    err = InitializationError("NotInitialized", "user", "Client has not been initialized.")
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def _build_wrapped_auth_error():
    """Reproduces the FLYTE-SDK-2A/2P chain shape: auth error wrapped twice."""
    from flyte.errors import RuntimeSystemError
    from flyte.remote._client.auth.errors import AuthenticationError

    try:
        try:
            try:
                raise AuthenticationError("Status Code (400) received from IDP: device code has expired.")
            except AuthenticationError as auth_err:
                raise RuntimeError(f"SelectCluster failed for operation=1: {auth_err}") from auth_err
        except RuntimeError:
            raise RuntimeSystemError("RuntimeError", "Failed to get signed url for /tmp/x.tar.gz.")
    except RuntimeSystemError as e:
        return e


def test_capture_exception_skips_wrapped_auth_error():
    err = _build_wrapped_auth_error()
    # Sanity-check we built the chain we expected (RuntimeSystemError -> RuntimeError -> AuthenticationError).
    from flyte.remote._client.auth.errors import AuthenticationError

    chain = list(_sentry._iter_cause_chain(err))
    assert any(isinstance(c, AuthenticationError) for c in chain)

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_wrapped_access_token_not_found_error():
    from flyte.errors import RuntimeSystemError
    from flyte.remote._client.auth.errors import AccessTokenNotFoundError

    try:
        try:
            try:
                raise AccessTokenNotFoundError("refresh token expired")
            except AccessTokenNotFoundError as auth_err:
                raise RuntimeError(f"SelectCluster failed: {auth_err}") from auth_err
        except RuntimeError:
            raise RuntimeSystemError("RuntimeError", "Failed to get signed url for /tmp/x.tar.gz.")
    except RuntimeSystemError as e:
        err = e

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_wrapped_deployment_error_via_cause_chain():
    """Even when a DeploymentError is wrapped in a plain RuntimeError, we filter."""
    from flyte.errors import DeploymentError

    try:
        try:
            raise DeploymentError("bad trigger config")
        except DeploymentError as dep_err:
            raise RuntimeError("outer wrapper") from dep_err
    except RuntimeError as e:
        err = e

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_still_reports_unrelated_runtime_errors():
    """An unrelated RuntimeError (no auth/user cause) should still go to Sentry."""
    err = RuntimeError("genuine SDK crash")
    with (
        mock.patch.object(_sentry, "init"),
        mock.patch("sentry_sdk.is_initialized", return_value=True),
        mock.patch("sentry_sdk.capture_exception") as capture_mock,
        mock.patch("sentry_sdk.flush"),
    ):
        _sentry.capture_exception(err)
    capture_mock.assert_called_once_with(err)


def test_iter_cause_chain_is_cycle_safe():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a  # cycle
    walked = list(_sentry._iter_cause_chain(a))
    assert walked == [a, b]


def test_capture_exception_skips_module_load_error():
    """ModuleLoadError inherits from RuntimeUserError and should be filtered.

    Reproduces FLYTE-SDK-3T/3K/3R/3Q/3P/3M/3N/3J/3H/3E: bare ModuleNotFoundError
    raised from user workflow imports is now wrapped as ModuleLoadError before
    reaching the Sentry boundary.
    """
    from flyte.errors import ModuleLoadError

    err = ModuleLoadError("Failed to load workflow.py: ModuleNotFoundError: No module named 'requests'")
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_runtime_user_error_subclass():
    """Any RuntimeUserError subclass is a user-side error, not an SDK crash."""
    from flyte.errors import OOMError

    err = OOMError("OOM", "user", "out of memory")
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_wrapped_module_load_error_via_cause_chain():
    """ModuleLoadError wrapped inside another exception (e.g. when re-raised
    deeper in the deploy path) is still filtered via __cause__ walking."""
    from flyte.errors import ModuleLoadError

    try:
        try:
            raise ModuleLoadError("Failed to load workflow.py: ModuleNotFoundError: No module named 'boto3'")
        except ModuleLoadError as e:
            raise RuntimeError("deploy failed") from e
    except RuntimeError as outer:
        err = outer

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_connect_error_unauthenticated():
    """FLYTE-SDK-33: ConnectError(Unauthenticated) from auth interceptor is a user
    credentials issue (expired token, IDP policy denial), not an SDK crash."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    err = ConnectError(
        Code.UNAUTHENTICATED,
        'transport: per-RPC creds failed due to error: failed to get new token: oauth2: "access_denied"',
    )
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_connect_error_permission_denied_wrapped_as_system_error():
    """FLYTE-SDK-40: cross-org call rejected with PermissionDenied is wrapped as
    RuntimeError("SelectCluster failed...") -> RuntimeSystemError("Failed to get signed url").
    The outer types are SDK errors, but the cause chain reveals a user config mistake."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from flyte.errors import RuntimeSystemError

    try:
        try:
            try:
                raise ConnectError(
                    Code.PERMISSION_DENIED,
                    "cross org calls are not allowed for organization [demo] on behalf of [default]",
                )
            except ConnectError as ce:
                raise RuntimeError(f"SelectCluster failed for operation=1: {ce}") from ce
        except RuntimeError:
            raise RuntimeSystemError("RuntimeError", "Failed to get signed url for /tmp/x.tar.gz.")
    except RuntimeSystemError as e:
        err = e

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_connect_error_failed_precondition_wrapped_as_system_error():
    """FLYTE-SDK-3S: backend returns FailedPrecondition ('no enabled clusters for org X')
    which surfaces as RuntimeSystemError('Failed to create run: ...'). Org config problem,
    not an SDK bug."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from flyte.errors import RuntimeSystemError

    try:
        try:
            raise ConnectError(Code.FAILED_PRECONDITION, "no enabled clusters found for org union-nav")
        except ConnectError:
            raise RuntimeSystemError(
                "RuntimeError", "Failed to create run: no enabled clusters found for org union-nav"
            )
    except RuntimeSystemError as e:
        err = e

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_still_reports_connect_error_internal():
    """ConnectError(INTERNAL/UNKNOWN) can indicate a real backend or SDK bug
    and should still be reported. UNAVAILABLE / DEADLINE_EXCEEDED are filtered
    separately (transient infra)."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    err = ConnectError(Code.INTERNAL, "backend panicked")
    with (
        mock.patch.object(_sentry, "init"),
        mock.patch("sentry_sdk.is_initialized", return_value=True),
        mock.patch("sentry_sdk.capture_exception") as capture_mock,
        mock.patch("sentry_sdk.flush"),
    ):
        _sentry.capture_exception(err)
    capture_mock.assert_called_once_with(err)


def test_capture_exception_skips_oserror_no_space_left():
    """FLYTE-SDK-32: OSError(ENOSPC) from shutil._fastcopy_sendfile during
    `flyte deploy` bundle upload is a user environment problem (disk full),
    not an SDK bug."""
    err = OSError(errno.ENOSPC, "No space left on device", "/some/path")
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_still_reports_other_oserror():
    """OSError with errnos other than ENOSPC may legitimately indicate SDK bugs
    and should still be reported to Sentry."""
    err = PermissionError(errno.EACCES, "Permission denied", "/some/path")
    with (
        mock.patch.object(_sentry, "init"),
        mock.patch("sentry_sdk.is_initialized", return_value=True),
        mock.patch("sentry_sdk.capture_exception") as capture_mock,
        mock.patch("sentry_sdk.flush"),
    ):
        _sentry.capture_exception(err)
    capture_mock.assert_called_once_with(err)


def test_capture_exception_skips_connect_error_unavailable_wrapped_as_system_error():
    """FLYTE-SDK-47/48/3W: SelectCluster transient network failures (Connection
    refused / reset / DNS lookup failed) surface as ConnectError(UNAVAILABLE)
    wrapped through RuntimeError -> RuntimeSystemError. These are infra
    problems, not SDK bugs."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from flyte.errors import RuntimeSystemError

    try:
        try:
            try:
                raise ConnectError(
                    Code.UNAVAILABLE,
                    "Request failed: error sending request for url (...): client error (Connect): "
                    "tcp connect error: Connection refused",
                )
            except ConnectError as ce:
                raise RuntimeError(f"SelectCluster failed for operation=1: {ce}") from ce
        except RuntimeError:
            raise RuntimeSystemError(
                "RuntimeError", "Failed to get signed url for /tmp/x.tar.gz: SelectCluster failed..."
            )
    except RuntimeSystemError as e:
        err = e

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_connect_error_deadline_exceeded_wrapped_as_system_error():
    """FLYTE-SDK-29: SelectCluster Request timed out surfaces as
    ConnectError(DEADLINE_EXCEEDED) wrapped through RuntimeError ->
    RuntimeSystemError. Transient infra, not an SDK bug."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from flyte.errors import RuntimeSystemError

    try:
        try:
            try:
                raise ConnectError(Code.DEADLINE_EXCEEDED, "Request timed out")
            except ConnectError as ce:
                raise RuntimeError(f"SelectCluster failed for operation=1: {ce}") from ce
        except RuntimeError:
            raise RuntimeSystemError("RuntimeError", "Failed to get signed url for /tmp/x.pb: SelectCluster failed...")
    except RuntimeSystemError as e:
        err = e

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_connect_error_unimplemented_wrapped_as_system_error():
    """FLYTE-SDK-4F: a raw HTTP 404 from the control-plane ingress surfaces as
    ConnectError(UNIMPLEMENTED, "Not Found") (connect maps 404 → UNIMPLEMENTED,
    not NOT_FOUND) wrapped through RuntimeError("SelectCluster failed...") ->
    RuntimeSystemError("Upload failed..."). The endpoint is wrong or the backend
    doesn't serve that RPC — not an SDK bug, so it shouldn't be crash-reported."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from flyte.errors import RuntimeSystemError

    try:
        try:
            try:
                raise ConnectError(Code.UNIMPLEMENTED, "Not Found")
            except ConnectError as ce:
                raise RuntimeError(f"SelectCluster failed for operation=1: {ce}") from ce
        except RuntimeError:
            raise RuntimeSystemError("RuntimeError", "Upload failed for /tmp/x/spec.pb (org='apple', ...): Not Found")
    except RuntimeSystemError as e:
        err = e

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def _wrap_as_upload_system_error(inner: BaseException):
    """Reproduce the flyte.remote._data shape: the real network failure is wrapped
    in RuntimeError('SelectCluster failed...') -> RuntimeSystemError('Failed to
    get signed url...'), so isinstance() on the outer exc misses the cause."""
    from flyte.errors import RuntimeSystemError

    try:
        try:
            try:
                raise inner
            except BaseException as net_err:
                raise RuntimeError(f"SelectCluster failed for operation=1: {net_err}") from net_err
        except RuntimeError:
            raise RuntimeSystemError("RuntimeError", "Failed to get signed url for /tmp/x.tar.gz.")
    except RuntimeSystemError as e:
        return e


def test_capture_exception_skips_timeout_error():
    """FLYTE-SDK-29: SelectCluster request times out (``TimeoutError``) on the way
    to the cluster service. A network/backend timeout is not an SDK crash."""
    err = _wrap_as_upload_system_error(TimeoutError("Request timed out"))
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_skips_connection_error():
    """FLYTE-SDK-47: builtin ``ConnectionError`` (connection refused) reaching the
    cluster service is a local network problem, not an SDK bug."""
    err = _wrap_as_upload_system_error(ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"))
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


@pytest.mark.parametrize(
    "httpx_exc_name", ["WriteError", "ReadError", "ConnectError", "ConnectTimeout", "RemoteProtocolError"]
)
def test_capture_exception_skips_httpx_transport_errors(httpx_exc_name):
    """FLYTE-SDK-3W / FLYTE-SDK-36 / FLYTE-SDK-4M: the signed-URL PUT fails at the
    transport layer (connection reset, read/write error, connect timeout) or the
    server hangs up mid-response (RemoteProtocolError). Transient network."""
    import httpx

    err = _wrap_as_upload_system_error(getattr(httpx, httpx_exc_name)("boom"))
    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()


def test_capture_exception_still_reports_connect_error_internal_in_upload_chain():
    """ConnectError(INTERNAL) — a backend 500 (FLYTE-SDK-43) — is intentionally NOT
    treated as transient: it can be a real backend bug, so it still reaches Sentry."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    err = _wrap_as_upload_system_error(ConnectError(Code.INTERNAL, "Internal Server Error"))
    with (
        mock.patch.object(_sentry, "init"),
        mock.patch("sentry_sdk.is_initialized", return_value=True),
        mock.patch("sentry_sdk.capture_exception") as capture_mock,
        mock.patch("sentry_sdk.flush"),
    ):
        _sentry.capture_exception(err)
    capture_mock.assert_called_once_with(err)


def test_capture_exception_skips_wrapped_invalid_endpoint_error():
    """FLYTE-SDK-5N: the auth-config endpoint returns HTML instead of protobuf, surfaced by
    connectrpc as ConnectError(UNKNOWN, 'invalid content-type ...'). RemoteClientConfigStore
    re-raises it as a user-facing InitializationError, which then gets wrapped as
    RuntimeSystemError('Upload failed ...') during a run. The cause chain reveals the user
    misconfiguration, so it must not be reported to Sentry."""
    from flyte.errors import InitializationError, RuntimeSystemError

    try:
        try:
            raise InitializationError(
                "InvalidEndpoint",
                "user",
                "The configured endpoint returned a non-protobuf (HTML) response ...",
            )
        except InitializationError:
            raise RuntimeSystemError("RuntimeError", "Upload failed for /tmp/spec.pb (org='x', ...).")
    except RuntimeSystemError as e:
        err = e

    chain = list(_sentry._iter_cause_chain(err))
    assert any(isinstance(c, InitializationError) for c in chain)

    with mock.patch.object(_sentry, "init") as init_mock:
        _sentry.capture_exception(err)
    init_mock.assert_not_called()
