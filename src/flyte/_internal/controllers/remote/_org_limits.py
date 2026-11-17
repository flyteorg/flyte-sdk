"""Recognizing a request the server refused because of an org limit or entitlement.

The server attaches a `cloudidl.common.OrgLimitExceeded` detail when a request would push the org past one of its
limits, and a `cloudidl.common.OrgEntitlementRequired` detail when the org isn't entitled to a capability. The SDK has
no cloudidl protos, so it decodes both with descriptors of its own (`_DETAILS`), recognizing them by type name. A
server older than the detail, or one that sends it without a field, falls back to the server's message.
"""

from __future__ import annotations

from typing import Any

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf import descriptor_pool, message_factory
from google.protobuf.descriptor_pb2 import DescriptorProto, FieldDescriptorProto, FileDescriptorProto
from google.protobuf.message import DecodeError, Message

import flyte.errors
from flyte.remote._client._connect_compat import detail_bytes, detail_type_name

ORG_LIMIT_EXCEEDED_TYPE = "cloudidl.common.OrgLimitExceeded"
ORG_ENTITLEMENT_REQUIRED_TYPE = "cloudidl.common.OrgEntitlementRequired"

# Sent on every enqueue: the controller fails a refusal carrying OrgLimitExceeded as a user error, without retrying,
# so the backend may refuse a child of a run over an org limit before storing it (otherwise it stores the child and
# fails it later). It describes this controller, not the SDK that built the task.
CAPABILITIES_HEADER = "x-actions-capabilities"
CAPABILITY_ORG_LIMIT_EXCEEDED = "org-limit-exceeded"

# The Rust controller hands errors to Python as strings; its ControllerError::OrgLimitExceeded displays with this
# prefix (rs_controller/src/error.rs).
RUST_ORG_LIMIT_PREFIX = "Org limit exceeded: "


def _details_pool() -> descriptor_pool.DescriptorPool:
    """The two detail messages, as defined in cloud's idl/common/error.proto, in a pool of their own so they can't
    clash with a cloudidl module a plugin registers in the default pool. Fields are matched by number, so a field
    the server adds later is skipped, and one it doesn't send reads as its default."""
    f = FieldDescriptorProto
    fdp = FileDescriptorProto(name="flyte/_org_limit_details.proto", package="cloudidl.common")
    fdp.syntax = "proto3"

    def add(msg: DescriptorProto, name: str, number: int, typ: int, repeated: bool = False):
        label = f.LABEL_REPEATED if repeated else f.LABEL_OPTIONAL
        msg.field.add(name=name, number=number, type=typ, label=label)

    limit = fdp.message_type.add(name="OrgLimitExceeded")
    add(limit, "limit", 1, f.TYPE_STRING)
    add(limit, "ceiling", 2, f.TYPE_INT64)
    add(limit, "current", 3, f.TYPE_INT64)
    add(limit, "requested", 4, f.TYPE_INT64)
    add(limit, "tier_default", 5, f.TYPE_INT64)
    limit.field[-1].proto3_optional = True
    limit.field[-1].oneof_index = 0
    limit.oneof_decl.add(name="_tier_default")
    add(limit, "upgrade_url", 6, f.TYPE_STRING)

    ent = fdp.message_type.add(name="OrgEntitlementRequired")
    add(ent, "entitlement", 1, f.TYPE_STRING)
    add(ent, "values", 2, f.TYPE_STRING, repeated=True)
    add(ent, "upgrade_url", 3, f.TYPE_STRING)

    pool = descriptor_pool.DescriptorPool()  # ty: ignore[possibly-missing-implicit-call]
    pool.Add(fdp)
    return pool


_POOL = _details_pool()
_DETAILS = {
    name: message_factory.GetMessageClass(_POOL.FindMessageTypeByName(name))
    for name in (ORG_LIMIT_EXCEEDED_TYPE, ORG_ENTITLEMENT_REQUIRED_TYPE)
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
    d: Any = _detail(e, ORG_LIMIT_EXCEEDED_TYPE)
    if d is None or not d.limit:
        return None
    return flyte.errors.OrgLimit(
        limit=d.limit,
        ceiling=d.ceiling,
        current=d.current,
        requested=d.requested,
        tier_default=d.tier_default if d.HasField("tier_default") else None,
        upgrade_url=d.upgrade_url,
    )


def _has_detail(e: ConnectError, type_name: str) -> bool:
    return any(detail_type_name(d) == type_name for d in e.details)


def describe_limit(limit: flyte.errors.OrgLimit, *, retry_later: bool) -> str:
    """A refusal at an org limit, for a person: the limit, the org's value and usage, and what to do about it.

    For example: "Org limit max_clusters (4) reached: 4 in use, 1 more requested. Its tier default is 2. Contact Union
    to raise it: https://www.union.ai/contact"
    """
    msg = f"Org limit {limit.limit} ({limit.ceiling}) reached: {limit.current} in use"
    msg += f", {limit.requested} more requested." if limit.requested else "."
    if limit.tier_default is not None and limit.tier_default != limit.ceiling:
        msg += f" Its tier default is {limit.tier_default}."
    action = (
        "Wait for running work to finish, or contact Union to raise it" if retry_later else "Contact Union to raise it"
    )
    return f"{msg} {action}: {limit.upgrade_url}" if limit.upgrade_url else f"{msg} {action}."


def describe_entitlement(entitlement: str, values: list[str], upgrade_url: str) -> str:
    """A refusal for a missing entitlement, for a person, e.g. "This org isn't entitled to allowed_clouds (coreweave).
    Contact Union to enable it: https://www.union.ai/contact"."""
    msg = f"This org isn't entitled to {entitlement}"
    if values:
        msg += f" ({', '.join(values)})"
    return f"{msg}. Contact Union to enable it: {upgrade_url}" if upgrade_url else f"{msg}. Contact Union to enable it."


def org_limit_exceeded(e: ConnectError) -> flyte.errors.OrgLimitExceededError | None:
    """Return the user error for e when it is an org limit refusal, else None. Only FAILED_PRECONDITION is one:
    the same detail with RESOURCE_EXHAUSTED (max_concurrent_runs) means "later", not "never" (see
    resource_exhausted)."""
    if e.code != Code.FAILED_PRECONDITION or not _has_detail(e, ORG_LIMIT_EXCEEDED_TYPE):
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
    d: Any = _detail(e, ORG_ENTITLEMENT_REQUIRED_TYPE)
    if d is None or not d.entitlement:
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
