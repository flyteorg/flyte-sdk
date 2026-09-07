"""Conformance harness — enforce the common provider format.

Every `flyteplugins.webhooks.<product>` plugin ships a one-line test:

```python
from flyte.extras.webhooks.testing import assert_provider_conforms
import flyteplugins.github as plugin


def test_conformance():
    assert_provider_conforms(plugin)
```

CI then fails if a plugin drifts from the shared format. The checks are the ones
that actually bit the per-product plugins this family replaces: verification
that raises instead of returning False, event constants that render as enum
names, dedupe keys that collapse distinct events onto one key.
"""

from __future__ import annotations

import typing

from ._event import WebhookEvent
from ._event_type import EventType
from ._provider import Provider


class ProviderFactory(typing.Protocol):
    """What a plugin's exported provider class must look like.

    Constructible with no arguments — the defaults are pre-wired — and
    accepting `secret_env` for anyone storing the secret under another name.
    """

    default_secret_env: str

    def __call__(self, *, secret_env: str | None = ...) -> Provider: ...


def assert_provider_conforms(plugin: typing.Any) -> None:
    """Assert that a provider plugin implements the common webhook contract.

    The contract:

    1. exports exactly one `Provider` subclass, constructible with no arguments
       and whose `name` matches the plugin's module name, so `/webhook/<name>`
       routes to it and `providers=[SomethingProvider()]` is all a user writes;
    2. exports an `events` module whose `__all__` names `EventType` subclasses,
       every member of which is a `str` that renders as its wire value;
    3. exports `SAMPLE_DELIVERY`, a `(headers, body)` pair of a real payload,
       which must verify under a known secret, reject the wrong secret, survive
       hostile headers without raising, parse into a `WebhookEvent` carrying
       this provider's name, and dedupe stably.

    Providers with `signed=True` must additionally reject a tampered body, since
    a signature covers it. A shared token does not, so `signed=False` opts out of
    that one check — and makes the dashboard say the product does not sign.

    `SAMPLE_DELIVERY` is what makes the rest checkable: without a real payload
    there is no way to assert that `parse` and `verify` actually agree with the
    product, and every check here would be vacuous.

    Raises `AssertionError` with a specific message on any deviation.
    """
    name = getattr(plugin, "__name__", repr(plugin))
    short = name.rsplit(".", 1)[-1]

    provider_class = _exported_provider_class(plugin, name)
    class_name = getattr(provider_class, "__name__", repr(provider_class))
    # Constructing with no arguments is the contract: defaults are pre-wired, so
    # `providers=[GitHubProvider()]` is all a user writes.
    try:
        provider = provider_class()
    except TypeError as exc:
        raise AssertionError(
            f"{name}: {class_name}() must construct with no arguments — "
            f"pre-wire the defaults in __init__ instead. Got: {exc}"
        ) from exc
    assert isinstance(provider, Provider), f"{name}: {class_name}() must be a Provider"
    assert provider.name == short, f"{name}: provider.name is {provider.name!r}, expected {short!r}"
    assert provider.secret_env, f"{name}: provider.secret_env must name an environment variable"
    assert provider_class.default_secret_env, (  # type: ignore[attr-defined]
        f"{name}: {class_name} must declare default_secret_env; WebhookAppEnvironment mounts it, "
        "so a plugin that omits it silently gives users an app with no secret"
    )
    assert provider.secret_env == provider_class.default_secret_env, (  # type: ignore[attr-defined]
        f"{name}: {class_name}() should default secret_env to default_secret_env"
    )
    assert callable(provider.verify) and callable(provider.parse), f"{name}: verify and parse must be callable"

    # The secret env var must be overridable, for users who store it elsewhere.
    custom = provider_class(secret_env="CONFORMANCE_OVERRIDE")
    assert custom.secret_env == "CONFORMANCE_OVERRIDE", (
        f"{name}: {class_name}(secret_env=...) must override the default"
    )

    _assert_events_conform(plugin, name)
    _assert_sample_delivery_conforms(plugin, provider, name)


def _exported_provider_class(plugin: typing.Any, name: str) -> ProviderFactory:
    """Find the one `Provider` subclass a plugin exports.

    Discovered rather than looked up by a fixed name, because each plugin names
    its own after the product (`GitHubProvider`, `SlackProvider`, ...).
    """
    exported = getattr(plugin, "__all__", None)
    assert exported, f"{name}: must define __all__"
    found = [
        obj
        for attr in exported
        if isinstance(obj := getattr(plugin, attr, None), type) and issubclass(obj, Provider) and obj is not Provider
    ]
    assert found, (
        f"{name}: must export a Provider subclass with its defaults pre-wired, "
        "so users can write providers=[SomethingProvider()]"
    )
    assert len(found) == 1, f"{name}: exports more than one Provider subclass: {[c.__name__ for c in found]}"
    return typing.cast(ProviderFactory, found[0])


def _assert_events_conform(plugin: typing.Any, name: str) -> None:
    events = getattr(plugin, "events", None)
    assert events is not None, f"{name}: must export an `events` module of typed constants"
    exported = getattr(events, "__all__", None)
    assert exported, f"{name}.events: must define __all__"

    seen: dict[str, str] = {}
    for cls_name in exported:
        cls = getattr(events, cls_name)
        assert isinstance(cls, type) and issubclass(cls, EventType), (
            f"{name}.events.{cls_name}: must subclass flyte.extras.webhooks.EventType"
        )
        assert len(cls) > 0, f"{name}.events.{cls_name}: has no members"
        for member in cls:
            assert isinstance(member, str), f"{name}.events.{cls_name}.{member.name}: must be a str"
            # 3.11+ would otherwise render "Class.MEMBER" into the dashboard.
            assert str(member) == member.value, (
                f"{name}.events.{cls_name}.{member.name}: str() gives {str(member)!r}, "
                f"expected the wire value {member.value!r} — subclass EventType, not str+Enum"
            )
            assert f"{member}" == member.value, f"{name}.events.{cls_name}.{member.name}: bad __format__"
            assert member.value not in seen, (
                f"{name}.events: {member.value!r} appears in both {seen[member.value]} and {cls_name}"
            )
            seen[member.value] = cls_name

        if "ANY" in cls.__members__:
            bare = cls.__members__["ANY"].value
            assert "." not in bare, f"{name}.events.{cls_name}.ANY should be a bare event type, got {bare!r}"
            for member in cls:
                if member is not cls.__members__["ANY"]:
                    assert member.value.startswith(f"{bare}."), (
                        f"{name}.events.{cls_name}.{member.name} ({member.value!r}) is not an action of {bare!r}"
                    )


def _assert_sample_delivery_conforms(plugin: typing.Any, provider: Provider, name: str) -> None:
    sample = getattr(plugin, "SAMPLE_DELIVERY", None)
    assert sample is not None, (
        f"{name}: must export SAMPLE_DELIVERY, a (headers, body) pair of a real payload. "
        "Without one there is no way to check that verify and parse agree with the product."
    )
    build_headers, body = sample
    assert callable(build_headers), f"{name}.SAMPLE_DELIVERY: first item must build headers from (body, secret)"
    assert isinstance(body, bytes), f"{name}.SAMPLE_DELIVERY: second item must be the raw body as bytes"

    secret = "conformance-secret"
    headers = build_headers(body, secret)

    assert provider.verify(body, headers, secret) is True, f"{name}: SAMPLE_DELIVERY does not verify under its secret"
    assert provider.verify(body, headers, "wrong-secret") is False, f"{name}: verify accepted the wrong secret"

    if provider.signed:
        # A signature covers the body, so tampering with it must fail. A shared
        # token does not, which is exactly why `signed=False` exists: the
        # dashboard says so, and this check does not pretend otherwise.
        assert provider.verify(b'{"tampered": true}', headers, secret) is False, (
            f"{name}: verify accepted a body that does not match its signature. "
            "If this product does not sign its webhooks, set signed=False on the Provider."
        )

    # Attacker-controlled headers must never raise. A non-ASCII credential is the
    # case that turns a clean 401 into a 500 when compared as str.
    for key, value in headers.items():
        if key.lower() not in _CREDENTIAL_HEADERS:
            continue
        for hostile in _hostile_variants(value):
            replaced = {**headers, key: hostile}
            try:
                accepted = provider.verify(body, replaced, secret)
            except Exception as exc:  # any raise at all is the failure, so catch broadly
                raise AssertionError(
                    f"{name}: verify raised {type(exc).__name__} on the hostile header "
                    f"{key}={hostile!r}: {exc}. Compare credentials as bytes — "
                    "flyte.extras.webhooks.constant_time_equals does this."
                ) from exc
            assert accepted is False, f"{name}: verify accepted {key}={hostile!r}"

    event = provider.parse(headers, body)
    assert isinstance(event, WebhookEvent), f"{name}: parse must return a WebhookEvent"
    assert event.provider == provider.name, (
        f"{name}: parsed event.provider is {event.provider!r}, expected {provider.name!r}"
    )
    assert event.event_type, f"{name}: parsed event has no event_type"
    assert event.qualified_type, f"{name}: parsed event has no qualified_type"

    key = event.dedupe_key()
    assert key and key == provider.parse(headers, body).dedupe_key(), (
        f"{name}: dedupe_key is not stable across two parses of the same delivery"
    )

    values = {m.value for cls_name in plugin.events.__all__ for m in getattr(plugin.events, cls_name)}
    assert event.qualified_type in values, (
        f"{name}: the sample delivery parses to {event.qualified_type!r}, which no constant in {name}.events spells"
    )


#: Headers that carry a credential, swapped for hostile values during conformance.
_CREDENTIAL_HEADERS = frozenset(
    {
        "x-hub-signature-256",
        "x-slack-signature",
        "x-linear-signature",
        "x-clickup-signature",
        "x-webhook-token",
    }
)


def _hostile_variants(value: str) -> list[str]:
    """Credential values an attacker could send, shaped to reach the comparison.

    A bare "\xff\xfe" is rejected by a provider's format check (GitHub wants a
    `sha256=` prefix, Slack a `v0=`) before it ever reaches `compare_digest` —
    which would make this check vacuous for exactly the providers it matters
    most for. So keep whatever scheme prefix the real header has and corrupt
    only the credential after it.
    """
    prefix, separator, _ = value.rpartition("=")
    head = f"{prefix}{separator}" if separator else ""
    return [f"{head}\xff\xfe", f"{head}\u00e9\u00e9", head or "\xff\xfe", "", "not-a-credential"]
