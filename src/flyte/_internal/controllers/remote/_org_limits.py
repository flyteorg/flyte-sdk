"""Recognizing a launch the server refused because the org is at one of its limits.

The server refuses with FAILED_PRECONDITION carrying a `cloudidl.common.OrgLimitExceeded` detail. The SDK has no
cloudidl protos, so the detail is recognized by its type URL; the error message already names the limit and the
numbers.
"""

from __future__ import annotations

from connectrpc.errors import ConnectError

import flyte.errors

ORG_LIMIT_EXCEEDED_TYPE = "cloudidl.common.OrgLimitExceeded"

# Sent on every enqueue: the controller fails a refusal carrying OrgLimitExceeded as a user error, without retrying,
# so the backend may refuse a child of a run over an org limit before storing it (otherwise it stores the child and
# fails it later). It describes this controller, not the SDK that built the task.
CAPABILITIES_HEADER = "x-actions-capabilities"
CAPABILITY_ORG_LIMIT_EXCEEDED = "org-limit-exceeded"

# The Rust controller hands errors to Python as strings; its ControllerError::OrgLimitExceeded displays with this
# prefix (rs_controller/src/error.rs).
RUST_ORG_LIMIT_PREFIX = "Org limit exceeded: "


def org_limit_exceeded(e: ConnectError) -> flyte.errors.OrgLimitExceededError | None:
    """Return the user error for e when it is an org limit refusal, else None."""
    for detail in e.details:
        if detail.type_url.rsplit("/", 1)[-1] == ORG_LIMIT_EXCEEDED_TYPE:
            return flyte.errors.OrgLimitExceededError(e.message)
    return None


def rust_client_error(err: str) -> Exception:
    """Convert a Rust controller client error string: an org limit refusal is a user error, anything else a system
    error."""
    if err.startswith(RUST_ORG_LIMIT_PREFIX):
        return flyte.errors.OrgLimitExceededError(err[len(RUST_ORG_LIMIT_PREFIX) :])
    return flyte.errors.RuntimeSystemError("RustControllerError", err)
