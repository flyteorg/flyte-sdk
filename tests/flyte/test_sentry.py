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
