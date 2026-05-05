"""Selection-logic tests for ``create_remote_controller``.

These tests exercise the env-var-based switching between the Python and
Rust-backed RemoteController implementations introduced for the Rust
controller port. They do NOT require ``flyte_controller_base`` to be
installed — when the Rust path is exercised, a fake module is injected
into ``sys.modules`` so the factory's ``import flyte_controller_base``
probe succeeds and the captured ``__new__`` arguments can be asserted.

Covered cases
-------------
* default path (no env vars set) returns the Python RemoteController
* ``_F_USE_RUST_CONTROLLER=1`` selects the Rust controller when the
  wheel is importable
* ``_F_USE_RUST_CONTROLLER=1`` falls back to Python when the wheel is
  *not* importable (with a warning)
* ``_U_USE_ACTIONS=1`` forces the Python path even when the Rust opt-in
  is also set, because the Rust side does not yet support the unified
  Actions service
* explicit ``api_key`` is plumbed straight through to the Rust
  constructor (SessionConfig parity)
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

import pytest


# --- helpers --------------------------------------------------------------


class _FakeBaseController:
    """Stand-in for the PyO3 ``BaseController`` class.

    Records every kwarg passed to ``__new__`` so tests can assert that
    the factory plumbed ``endpoint`` / ``api_key`` / ``workers`` through
    correctly.
    """

    last_new_kwargs: dict[str, Any] = {}

    def __new__(cls, **kwargs: Any) -> "_FakeBaseController":
        _FakeBaseController.last_new_kwargs = dict(kwargs)
        return object.__new__(cls)

    def __init__(self, **kwargs: Any) -> None:  # accept and ignore
        pass


@pytest.fixture(autouse=True)
def reset_kwargs() -> None:
    _FakeBaseController.last_new_kwargs = {}


@pytest.fixture
def fake_rust_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a stub ``flyte_controller_base`` module so the factory's
    ``import flyte_controller_base`` probe succeeds and the Rust-backed
    ``_r_controller.RemoteController`` can subclass our fake base."""
    fake_module = type(sys)("flyte_controller_base")
    fake_module.Action = _FakeBaseController  # type: ignore[attr-defined]
    fake_module.BaseController = _FakeBaseController  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flyte_controller_base", fake_module)
    # Also drop any cached _r_controller import so it picks up the fake
    # base class on next import.
    monkeypatch.delitem(sys.modules, "flyte._internal.controllers.remote._r_controller", raising=False)


def _import_factory():
    """Import (or re-import) the factory after env vars / sys.modules
    have been monkeypatched, so module-level state reflects the test."""
    sys.modules.pop("flyte._internal.controllers.remote", None)
    from flyte._internal.controllers.remote import create_remote_controller

    return create_remote_controller


# --- tests ----------------------------------------------------------------


def test_default_path_selects_python_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("_F_USE_RUST_CONTROLLER", raising=False)
    monkeypatch.delenv("_U_USE_ACTIONS", raising=False)

    create_remote_controller = _import_factory()

    # Avoid actually constructing channels by patching the Python
    # RemoteController constructor and the ControllerClient factories.
    with (
        patch(
            "flyte._internal.controllers.remote._client.ControllerClient.for_endpoint"
        ) as mock_for_endpoint,
        patch("flyte._internal.controllers.remote._controller.RemoteController") as mock_py_rc,
    ):
        mock_for_endpoint.return_value = object()
        create_remote_controller(endpoint="dns:///example.com:443")

    mock_py_rc.assert_called_once()
    # The Rust constructor must NOT have been invoked.
    assert _FakeBaseController.last_new_kwargs == {}


def test_rust_opt_in_selects_rust_controller(
    monkeypatch: pytest.MonkeyPatch, fake_rust_wheel: None
) -> None:
    monkeypatch.setenv("_F_USE_RUST_CONTROLLER", "1")
    monkeypatch.delenv("_U_USE_ACTIONS", raising=False)

    create_remote_controller = _import_factory()
    controller = create_remote_controller(endpoint="dns:///example.com:443")

    # Rust path was taken — the fake BaseController saw the call.
    assert _FakeBaseController.last_new_kwargs.get("endpoint") == "dns:///example.com:443"
    assert _FakeBaseController.last_new_kwargs.get("api_key") is None

    # Returned object is the Rust-backed RemoteController.
    assert type(controller).__name__ == "RemoteController"
    assert type(controller).__module__ == "flyte._internal.controllers.remote._r_controller"


def test_api_key_is_plumbed_into_rust_constructor(
    monkeypatch: pytest.MonkeyPatch, fake_rust_wheel: None
) -> None:
    """Regression: the Rust controller used to read _UNION_EAGER_API_KEY
    from the environment instead of accepting an explicit api_key. The
    SessionConfig integration plumbed the key through __new__; verify."""
    monkeypatch.setenv("_F_USE_RUST_CONTROLLER", "1")
    monkeypatch.delenv("_U_USE_ACTIONS", raising=False)

    create_remote_controller = _import_factory()
    # Note: the value here is opaque to the factory — we're just checking
    # that whatever is supplied gets forwarded verbatim to BaseController.
    create_remote_controller(api_key="opaque-base64-payload")

    assert _FakeBaseController.last_new_kwargs.get("api_key") == "opaque-base64-payload"


def test_use_actions_forces_python_path_even_with_rust_opt_in(
    monkeypatch: pytest.MonkeyPatch, fake_rust_wheel: None
) -> None:
    """The Rust controller does not yet implement the unified Actions
    service; ``_U_USE_ACTIONS=1`` must force the Python controller even
    when the Rust opt-in is also set."""
    monkeypatch.setenv("_F_USE_RUST_CONTROLLER", "1")
    monkeypatch.setenv("_U_USE_ACTIONS", "1")

    create_remote_controller = _import_factory()

    with (
        patch(
            "flyte._internal.controllers.remote._client.ControllerClient.for_endpoint"
        ) as mock_for_endpoint,
        patch("flyte._internal.controllers.remote._controller.RemoteController") as mock_py_rc,
    ):
        mock_for_endpoint.return_value = object()
        create_remote_controller(endpoint="dns:///example.com:443")

    mock_py_rc.assert_called_once()
    assert _FakeBaseController.last_new_kwargs == {}


def test_rust_opt_in_falls_back_when_wheel_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the user opts in to the Rust path but the wheel is not
    importable, the factory must log a warning and fall back to the
    Python controller — never raise ImportError to the caller."""
    monkeypatch.setenv("_F_USE_RUST_CONTROLLER", "1")
    monkeypatch.delenv("_U_USE_ACTIONS", raising=False)
    # Make sure no real or stub wheel is on the path.
    monkeypatch.delitem(sys.modules, "flyte_controller_base", raising=False)
    monkeypatch.delitem(
        sys.modules, "flyte._internal.controllers.remote._r_controller", raising=False
    )

    # Block the import.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "flyte_controller_base":
            raise ImportError("simulated: flyte_controller_base is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    create_remote_controller = _import_factory()

    # The flyte logger has propagate=False and its own StreamHandler,
    # so caplog/capsys don't reliably catch it. Patch the warning method
    # on the logger that the factory module actually uses.
    with (
        patch(
            "flyte._internal.controllers.remote._client.ControllerClient.for_endpoint"
        ) as mock_for_endpoint,
        patch("flyte._internal.controllers.remote._controller.RemoteController") as mock_py_rc,
        patch("flyte._internal.controllers.remote.logger.warning") as mock_warning,
    ):
        mock_for_endpoint.return_value = object()
        create_remote_controller(endpoint="dns:///example.com:443")

    mock_py_rc.assert_called_once()
    # Identify the fallback warning by its distinctive substring.
    warning_messages = [str(call.args[0]) for call in mock_warning.call_args_list]
    assert any("flyte_controller_base is not importable" in m for m in warning_messages), (
        f"expected fallback warning, got: {warning_messages!r}"
    )
