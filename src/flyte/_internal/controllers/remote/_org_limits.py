"""Recognizing a request the server refused because of an org limit or entitlement.

The server attaches a `flyteidl2.org.OrgLimitExceeded` detail when a request would push the org past one of its
limits, and a `flyteidl2.org.OrgEntitlementRequired` detail when the org isn't entitled to a capability. A server from
before those details sends only `cloudidl.common.OrgLimitExceeded`, which the SDK recognises by type name and reports
with the server's message.
"""

from __future__ import annotations

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.org import limits_pb2
from google.protobuf.message import DecodeError, Message

import flyte.errors
from flyte.remote._client._connect_compat import detail_bytes, detail_type_name

ORG_LIMIT_EXCEEDED_TYPE = "flyteidl2.org.OrgLimitExceeded"
ORG_ENTITLEMENT_REQUIRED_TYPE = "flyteidl2.org.OrgEntitlementRequired"
# What a server from before the flyteidl2 details attaches to an org limit refusal (servers since attach it too).
LEGACY_ORG_LIMIT_EXCEEDED_TYPE = "cloudidl.common.OrgLimitExceeded"

# Sent on every enqueue: the controller fails a refusal carrying OrgLimitExceeded as a user error, without retrying,
# so the backend may refuse a child of a run over an org limit before storing it (otherwise it stores the child and
# fails it later). It describes this controller, not the SDK that built the task.
CAPABILITIES_HEADER = "x-actions-capabilities"
CAPABILITY_ORG_LIMIT_EXCEEDED = "org-limit-exceeded"

# The Rust controller hands errors to Python as strings; its ControllerError::OrgLimitExceeded displays with this
# prefix (rs_controller/src/error.rs).
RUST_ORG_LIMIT_PREFIX = "Org limit exceeded: "

_DETAILS: dict[str, type[Message]] = {
    ORG_LIMIT_EXCEEDED_TYPE: limits_pb2.OrgLimitExceeded,
    ORG_ENTITLEMENT_REQUIRED_TYPE: limits_pb2.OrgEntitlementRequired,
}


def _detail(e: ConnectError, type_name: str) -> Message | None:
    """The first detail of e of type_name, decoded; None when e has none or it doesn't decode."""
    for d in e.details:
        if detail_type_name(d) != type_name:
            continue
        msg = _DETAILS[type_name]()
        try:
            msg.ParseFromString(detail_bytes(d))
        except (DecodeError, AttributeError, TypeError):
            return None
        return msg
    return None


def _org_limit(e: ConnectError) -> flyte.errors.OrgLimit | None:
    """e's OrgLimitExceeded detail, or None when it has none, or one without a limit name."""
    d = _detail(e, ORG_LIMIT_EXCEEDED_TYPE)
    if not isinstance(d, limits_pb2.OrgLimitExceeded) or not d.limit:
        return None
    return flyte.errors.OrgLimit(
        limit=d.limit,
        ceiling=d.ceiling,
        current=d.current,
        requested=d.requested,
        tier_default=d.tier_default if d.HasField("tier_default") else None,
        upgrade_url=d.upgrade_url,
        tier=d.tier,
    )


def _has_detail(e: ConnectError, type_name: str) -> bool:
    return any(detail_type_name(d) == type_name for d in e.details)


def describe_limit(limit: flyte.errors.OrgLimit, *, retry_later: bool) -> str:
    """A refusal at an org limit, for a person: the limit, the org's value and usage, and what to do about it.

    For example: "Org limit max_clusters (4) reached: 4 in use, 1 more requested. The Teams tier default is 2.
    Contact sales to raise it: https://www.union.ai/get-a-quote"
    """
    msg = f"Org limit {limit.limit} ({limit.ceiling}) reached: {limit.current} in use"
    msg += f", {limit.requested} more requested." if limit.requested else "."
    if limit.tier_default is not None and limit.tier_default != limit.ceiling:
        tier = f"The {limit.tier} tier" if limit.tier else "The org's tier"
        msg += f" {tier} default is {limit.tier_default}."
    action = (
        "Wait for running work to finish, or contact sales to raise it" if retry_later else "Contact sales to raise it"
    )
    return f"{msg} {action}: {limit.upgrade_url}" if limit.upgrade_url else f"{msg} {action}."


def describe_entitlement(entitlement: str, values: list[str], upgrade_url: str) -> str:
    """A refusal for a missing entitlement, for a person, e.g. "This org isn't entitled to allowed_clouds (coreweave).
    Contact sales to enable it: https://www.union.ai/get-a-quote"."""
    msg = f"This org isn't entitled to {entitlement}"
    if values:
        msg += f" ({', '.join(values)})"
    return f"{msg}. Contact sales to enable it: {upgrade_url}" if upgrade_url else f"{msg}. Contact sales to enable it."


def org_limit_exceeded(e: ConnectError) -> flyte.errors.OrgLimitExceededError | None:
    """Return the user error for e when it is an org limit refusal, else None. Only FAILED_PRECONDITION is one:
    the same detail with RESOURCE_EXHAUSTED (max_concurrent_runs) means "later", not "never" (see
    resource_exhausted)."""
    if e.code != Code.FAILED_PRECONDITION or not (
        _has_detail(e, ORG_LIMIT_EXCEEDED_TYPE) or _has_detail(e, LEGACY_ORG_LIMIT_EXCEEDED_TYPE)
    ):
        return None
    limit = _org_limit(e)
    if limit is None:
        return flyte.errors.OrgLimitExceededError(e.message)
    return flyte.errors.OrgLimitExceededError(describe_limit(limit, retry_later=False), org_limit=limit)


def resource_exhausted(e: ConnectError, prefix: str = "") -> flyte.errors.ResourceExhaustedError | None:
    """Return the "retry later" user error for a RESOURCE_EXHAUSTED e, else None. With an OrgLimitExceeded detail
    (max_concurrent_runs, max_queue_depth) the message comes from it; otherwise it is prefix + the server's."""
    if e.code != Code.RESOURCE_EXHAUSTED:
        return None
    limit = _org_limit(e)
    if limit is None:
        return flyte.errors.ResourceExhaustedError(prefix + e.message)
    return flyte.errors.ResourceExhaustedError(describe_limit(limit, retry_later=True), org_limit=limit)


def org_entitlement_required(e: ConnectError) -> flyte.errors.OrgEntitlementRequiredError | None:
    """Return the user error for e when it is a PERMISSION_DENIED entitlement refusal, else None."""
    if e.code != Code.PERMISSION_DENIED:
        return None
    d = _detail(e, ORG_ENTITLEMENT_REQUIRED_TYPE)
    if not isinstance(d, limits_pb2.OrgEntitlementRequired) or not d.entitlement:
        return None
    values = list(d.values)
    return flyte.errors.OrgEntitlementRequiredError(
        describe_entitlement(d.entitlement, values, d.upgrade_url),
        entitlement=d.entitlement,
        values=values,
        upgrade_url=d.upgrade_url,
    )


def org_refusal(e: ConnectError) -> flyte.errors.RuntimeUserError | None:
    """The user error for e when it is any org limit or entitlement refusal that carries its detail, else None."""
    if e.code == Code.RESOURCE_EXHAUSTED and _org_limit(e) is None:
        return None
    return org_limit_exceeded(e) or resource_exhausted(e) or org_entitlement_required(e)


def rust_client_error(err: str) -> Exception:
    """Convert a Rust controller client error string: an org limit refusal is a user error, anything else a system
    error."""
    if err.startswith(RUST_ORG_LIMIT_PREFIX):
        return flyte.errors.OrgLimitExceededError(err[len(RUST_ORG_LIMIT_PREFIX) :])
    return flyte.errors.RuntimeSystemError("RustControllerError", err)
