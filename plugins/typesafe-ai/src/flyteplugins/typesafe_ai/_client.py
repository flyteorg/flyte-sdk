"""Reaching TypeSafe from a Flyte task, and saying so clearly when you can't."""

from __future__ import annotations

import os
from typing import Any, Optional

#: The environment variable the TypeSafe SDK reads. Mount your secret as this:
#:
#:     flyte.Secret(key="TYPESAFE_API_KEY", as_env_var="TYPESAFE_API_KEY")
#:
#: `flyte.Secret` derives `as_env_var` from the key by upper-casing it and
#: swapping `-` for `_`, so a key named TYPESAFE_API_KEY mounts correctly on its
#: own -- spelling it out just makes the env var greppable.
API_KEY_ENV = "TYPESAFE_API_KEY"


class MissingAPIKey(RuntimeError):
    """Raised at the point of use, in the task that actually needs the key."""


def _explain() -> str:
    return (
        f"{API_KEY_ENV} is not set, so System 1 cannot be reached.\n\n"
        "Declare the secret on the TaskEnvironment that runs this task:\n\n"
        "    import flyte\n\n"
        "    env = flyte.TaskEnvironment(\n"
        '        "triage",\n'
        f'        secrets=[flyte.Secret(key="{API_KEY_ENV}", as_env_var="{API_KEY_ENV}")],\n'
        "    )\n\n"
        "and create it once, if it does not exist yet:\n\n"
        f"    flyte create secret {API_KEY_ENV} --value <your key>\n\n"
        "The check happens here, at the point of use, rather than at import: a task that merely "
        "passes answers along does not need the key, and an import-time failure would take down "
        "every task in the module."
    )


def client(*, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs: Any):
    """An `AsyncTypeSafeClient`, or a message that says exactly what to do.

    Without this, a missing key surfaces as a 401 from inside the vendor SDK, which
    tells you nothing about Flyte secrets.
    """
    from typesafe_sdk import AsyncTypeSafeClient

    key = api_key or os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise MissingAPIKey(_explain())
    return AsyncTypeSafeClient(api_key=key, model=model, **kwargs)
