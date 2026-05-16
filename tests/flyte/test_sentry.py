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
