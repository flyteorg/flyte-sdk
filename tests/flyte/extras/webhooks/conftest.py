"""Fixtures for the webhook receiver tests."""

from __future__ import annotations

import pytest

from ._stub import STUB_SECRET


@pytest.fixture
def secrets(monkeypatch):
    """Mount the stub providers' secrets."""
    monkeypatch.setenv("STUB_WEBHOOK_SECRET", STUB_SECRET)
    monkeypatch.setenv("UNSIGNED_WEBHOOK_TOKEN", STUB_SECRET)
