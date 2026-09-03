"""Errors raised by the webhooks plugin."""

from __future__ import annotations


class WebhookPluginError(Exception):
    """Base class for all errors raised by this plugin."""


class SignatureError(WebhookPluginError):
    """Raised when an inbound delivery fails verification or cannot be parsed."""
