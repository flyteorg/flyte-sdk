from types import SimpleNamespace

from flyteplugins.trackio._decorator import _build_init_kwargs


def test_build_init_kwargs_empty(monkeypatch):
    monkeypatch.setattr(
        "flyteplugins.trackio._decorator.get_trackio_context",
        lambda: None,
    )

    assert _build_init_kwargs({}) == {}


def test_build_init_kwargs_from_context(monkeypatch):
    ctx = SimpleNamespace(
        to_trackio_init=lambda: {
            "project": "vision",
            "server_url": "https://trackio.example.com",
        }
    )

    monkeypatch.setattr(
        "flyteplugins.trackio._decorator.get_trackio_context",
        lambda: ctx,
    )

    assert _build_init_kwargs({}) == {
        "project": "vision",
        "server_url": "https://trackio.example.com",
    }


def test_build_init_kwargs_decorator_overrides_context(monkeypatch):
    ctx = SimpleNamespace(
        to_trackio_init=lambda: {
            "project": "vision",
            "server_url": "https://trackio.example.com",
        }
    )

    monkeypatch.setattr(
        "flyteplugins.trackio._decorator.get_trackio_context",
        lambda: ctx,
    )

    kwargs = _build_init_kwargs(
        {
            "project": "nlp",
        }
    )

    assert kwargs == {
        "project": "nlp",
        "server_url": "https://trackio.example.com",
    }


def test_build_init_kwargs_ignores_none(monkeypatch):
    ctx = SimpleNamespace(
        to_trackio_init=lambda: {
            "project": "vision",
            "server_url": "https://trackio.example.com",
        }
    )

    monkeypatch.setattr(
        "flyteplugins.trackio._decorator.get_trackio_context",
        lambda: ctx,
    )

    kwargs = _build_init_kwargs(
        {
            "project": None,
        }
    )

    assert kwargs == {
        "project": "vision",
        "server_url": "https://trackio.example.com",
    }


def test_build_init_kwargs_decorator_only(monkeypatch):
    monkeypatch.setattr(
        "flyteplugins.trackio._decorator.get_trackio_context",
        lambda: None,
    )

    kwargs = _build_init_kwargs(
        {
            "project": "vision",
            "space_id": "user/demo",
        }
    )

    assert kwargs == {
        "project": "vision",
        "space_id": "user/demo",
    }
