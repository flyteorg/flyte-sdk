"""MCP secret tools for `flyte.ai.mcp.FlyteMCPAppEnvironment`.

Split out of `_tools.py` for size. Same shape as there: each tool closes over the environment
(`env`) and its docstring is what the MCP client shows the model.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

import flyte
import flyte.remote
from flyte.ai.mcp._secret_input import (
    SECRET_INPUT_PATH,
    SecretOp,
    _already_exists_message,
    _cli_fallback,
    _client_supports_url_elicitation,
    _http_request,
    _missing_message,
    _public_base_url,
    _secret_exists,
    run_validator,
)
from flyte.ai.mcp._tools import MCPContext, ToolError, _ts

if TYPE_CHECKING:
    from flyte.ai.mcp._flyte_mcp_app import FlyteMCPAppEnvironment

#: Key kinds `generate_secret` can produce.
SSH_KEY_KINDS = ("ssh-ed25519", "ssh-rsa")


def build_secret_tools(env: FlyteMCPAppEnvironment) -> dict[str, Any]:
    """Build the secret tools bound to `env`, keyed by tool name, in registration order."""

    async def get_secret(
        name: str,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Check whether a secret exists and has propagated to the clusters that run tasks.

        Use this after `create_secret`/`update_secret` to see whether the secret is valid and
        ready: `ready` is true once every cluster has it. It reports status only, never the
        value, and cannot tell you whether the value is *correct*. To check that the content
        works, run a task that requests the secret with `run_task`. A secret that does not
        exist comes back with `exists: false` rather than an error. After `delete_secret` it can
        keep reporting `exists: true` for up to ~30 s.
        """
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
        from flyteidl2.secret import definition_pb2

        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            try:
                secret = await cast(Any, flyte.remote.Secret.get).aio(name)
            except ConnectError as e:
                if e.code != Code.NOT_FOUND:
                    raise
                return {
                    "name": name,
                    "project": resolved_project,
                    "domain": resolved_domain,
                    "exists": False,
                    "ready": False,
                }
            meta = secret.pb2.secret_metadata
            return {
                "name": secret.name,
                "project": resolved_project,
                "domain": resolved_domain,
                "exists": True,
                "type": secret.type,
                "created_time": _ts(meta, "created_time"),
                "overall_status": definition_pb2.OverallStatus.Name(meta.secret_status.overall_status),
                "cluster_status": [
                    {
                        "cluster": c.cluster.name,
                        "presence": definition_pb2.SecretPresenceStatus.Name(c.presence_status),
                    }
                    for c in meta.secret_status.cluster_status
                ],
                "ready": meta.secret_status.overall_status == definition_pb2.OverallStatus.FULLY_PRESENT,
            }

    def _check_secret_type(type: str) -> None:
        # Secret.create treats *anything* that is not "regular" as an image_pull secret, which
        # then fails deeper down on the project/domain scope this server always sets. Reject the
        # typo here so the caller gets a sentence they can act on.
        if type != "regular":
            raise ToolError(
                f"Unsupported secret type '{type}'. Only 'regular' secrets can be created through this "
                f"server; image_pull secrets are scoped to a cluster pool, not to a project/domain."
            )

    def _check_validator(validator: str | None) -> None:
        if validator is not None and validator not in (env.secret_validators or {}):
            raise ToolError(
                f"Unknown validator '{validator}'. Registered validators: {sorted(env.secret_validators or {})}."
            )

    async def _validate_inline(validator: str | None, value: str) -> None:
        _check_validator(validator)
        if validator is not None:
            assert env.secret_validators is not None  # _check_validator accepted the name
            rejection = await run_validator(env.secret_validators[validator], value)
            if rejection is not None:
                raise ToolError(f"The value was not saved. {rejection}")

    async def _check_exists(op: SecretOp, name: str) -> None:
        """Refuse a create over an existing secret or an update of a missing one. Call inside `env._scoped`.

        The control plane does neither: `Secret.create` silently replaces the value of an existing
        secret and `Secret.update` creates a missing one, so the check has to be ours. It is
        check-then-write, so another writer landing between the check and the write is not
        prevented; the control plane offers no create-if-absent.
        """
        exists = await _secret_exists(name)
        if op == "create" and exists:
            raise ToolError(_already_exists_message(name))
        if op == "update" and not exists:
            raise ToolError(_missing_message(name))

    async def _request_secret_value(
        op: SecretOp,
        name: str,
        project: str | None,
        domain: str | None,
        validator: str | None,
        ctx: MCPContext | None,
    ) -> dict:
        """Hand the human a one-time URL for entering the value, or point at the CLI.

        Returns the `awaiting_value` result, or raises `UrlElicitationRequiredError` when the
        client supports URL elicitation. The value itself is never an input or an output here.
        """
        _check_validator(validator)
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            await _check_exists(op, name)
            cli_fallback = _cli_fallback(name, resolved_project, resolved_domain)
            request = None if env.transport == "stdio" else _http_request(ctx)
            if request is None:
                return {
                    "status": "awaiting_value",
                    "op": op,
                    "name": name,
                    "project": resolved_project,
                    "domain": resolved_domain,
                    "cli_fallback": cli_fallback,
                    "message": (
                        "This connection has no web form for entering the value. Ask the human to run the "
                        "command in `cli_fallback` in their own terminal; it prompts for the value with "
                        "hidden input. Do NOT ask them to paste the value into this chat."
                        + (" The command does not run the validator, so the value is not checked." if validator else "")
                    ),
                }
            token, pending = env._secret_inputs.mint(op, name, resolved_project, resolved_domain, validator)

        url = f"{_public_base_url(request)}{SECRET_INPUT_PATH}/{token}"
        verb = "create" if op == "create" else "update"
        if _client_supports_url_elicitation(ctx):
            from mcp.shared.exceptions import UrlElicitationRequiredError
            from mcp.types import ElicitRequestURLParams

            raise UrlElicitationRequiredError(
                [
                    ElicitRequestURLParams(
                        mode="url",
                        url=url,
                        elicitationId=token,
                        message=f"Enter the value to {verb} secret '{name}' in {resolved_project}/{resolved_domain}.",
                    )
                ]
            )
        return {
            "status": "awaiting_value",
            "op": op,
            "name": name,
            "project": resolved_project,
            "domain": resolved_domain,
            "url": url,
            "expires_at": datetime.fromtimestamp(pending.expires_at, timezone.utc).isoformat(),
            "cli_fallback": cli_fallback,
            "message": (
                f"The secret has NOT been {verb}d yet. Give the human this URL and ask them to open it "
                f"and enter the value there. Do NOT ask them to paste the value into this chat. "
                f"The link works once and expires at `expires_at`. Afterwards, call `get_secret` to "
                f"confirm it exists and is ready."
            ),
        }

    async def create_secret(
        name: str,
        project: str | None = None,
        domain: str | None = None,
        validator: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Start creating a secret in a project/domain. You never see or send the value.

        This does not create anything yet. It returns a one-time URL (`status: awaiting_value`);
        give it to the human, who opens it and types the value into a password field. Never ask
        the human to paste the value into the chat. If the result carries only a `cli_fallback`
        command, ask the human to run it. Afterwards call `get_secret` to confirm the secret is
        ready. Fails without changing anything if the secret already exists (the control plane
        would silently replace its value); use `update_secret` to replace a value.
        """
        return await _request_secret_value("create", name, project, domain, validator, ctx)

    async def update_secret(
        name: str,
        project: str | None = None,
        domain: str | None = None,
        validator: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Start replacing the value of an existing secret. You never see or send the value.

        This does not change anything yet. It returns a one-time URL (`status: awaiting_value`);
        give it to the human, who opens it and types the new value into a password field. Never
        ask the human to paste the value into the chat. If the result carries only a
        `cli_fallback` command, ask the human to run it. Afterwards call `get_secret` to confirm
        the secret is ready; tasks pick up the new value when they next start. Fails without
        changing anything if the secret does not exist (the control plane would create it); use
        `create_secret` for that.
        """
        return await _request_secret_value("update", name, project, domain, validator, ctx)

    async def create_secret_inline(
        name: str,
        value: str,
        type: str = "regular",
        project: str | None = None,
        domain: str | None = None,
        validator: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Create a secret in a project/domain. The value is written, never echoed back.

        `type` must be "regular"; "image_pull" secrets are scoped to a cluster pool rather
        than to a project/domain and cannot be created here. Fails without changing anything if
        the secret already exists (the control plane would silently replace its value); use
        `update_secret` to replace a value.
        """
        _check_secret_type(type)
        await _validate_inline(validator, value)
        with env._scoped(project, domain):
            await _check_exists("create", name)
            await cast(Any, flyte.remote.Secret.create).aio(name=name, value=value, type=type)
            return {"name": name, "created": True}

    async def update_secret_inline(
        name: str,
        value: str,
        project: str | None = None,
        domain: str | None = None,
        validator: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Replace the value of an existing secret in a project/domain. The value is never echoed back.

        Fails without changing anything if the secret does not exist (the control plane would
        create it); use `create_secret` for that.
        """
        await _validate_inline(validator, value)
        with env._scoped(project, domain):
            await _check_exists("update", name)
            await cast(Any, flyte.remote.Secret.update).aio(name=name, value=value)
            return {"name": name, "updated": True}

    async def generate_secret(
        name: str,
        kind: str = "ssh-ed25519",
        comment: str | None = None,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Generate an SSH key pair on the server and store the private key as a secret.

        Use this when a task needs a key, for example a GitHub deploy key. The private key is
        stored as the secret `name` and is never shown to anyone; you get the public key and its
        fingerprint, which the human installs on the other side (for example as a deploy key).
        `kind` is "ssh-ed25519" (default) or "ssh-rsa" (4096-bit, for hosts that do not accept
        ed25519). `comment` is appended to the public key line; it defaults to `name`. This tool
        never overwrites: it checks first and fails, changing nothing, if the secret already
        exists. Pick another name, or `delete_secret` first if you mean to replace it.
        """
        if kind not in SSH_KEY_KINDS:
            raise ToolError(f"Unsupported key kind '{kind}'. Supported kinds: {list(SSH_KEY_KINDS)}.")
        comment = name if comment is None else comment
        # The comment becomes part of a single-line public key; anything else would corrupt it.
        if not comment.isprintable():
            raise ToolError("The comment must be a single line of printable characters.")
        # Imported here so the MCP module does not pull cryptography in at import time.
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
        except ImportError as e:
            raise ToolError("generate_secret needs the `cryptography` package, which is not installed.") from e

        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            if await _secret_exists(name):
                raise ToolError(
                    f"Secret '{name}' already exists, so no key was generated and nothing was changed. "
                    f"Pick another name, or `delete_secret` first if you mean to replace it."
                )
            private_key: ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey
            if kind == "ssh-ed25519":
                private_key = ed25519.Ed25519PrivateKey.generate()
            else:
                # 4096-bit RSA takes seconds; keep it off the event loop.
                private_key = await asyncio.to_thread(rsa.generate_private_key, 65537, 4096)
            private_pem = private_key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption()
            ).decode()
            public_key = private_key.public_key().public_bytes(
                serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
            )
            # OpenSSH fingerprint: SHA-256 of the decoded key blob, base64 without padding.
            blob = base64.b64decode(public_key.split()[1])
            fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
            try:
                await cast(Any, flyte.remote.Secret.create).aio(name=name, value=private_pem)
            except Exception as e:
                # Redact in case the error quotes the value. The existence check above is
                # check-then-write: a secret created by someone else in between would be replaced.
                reason = str(e).replace(private_pem, "[redacted]")
                raise ToolError(f"Could not create secret '{name}' ({type(e).__name__}: {reason}).") from None
        return {
            "name": name,
            "project": resolved_project,
            "domain": resolved_domain,
            "kind": kind,
            # An empty comment would leave a trailing space on the line.
            "public_key": " ".join(x for x in (public_key.decode(), comment) if x),
            "fingerprint": fingerprint,
            "created": True,
            "message": (
                "The key pair was generated on the server and the private half is stored as the secret; "
                "it is never shown. Give `public_key` to the human to install on the other side (for "
                "example as a GitHub deploy key) and compare `fingerprint` if they want to verify it."
            ),
        }

    def _with_validator_param(fn: Any) -> Any:
        """Offer `validator` only when the env registered some, and tell the model which.

        FastMCP builds the input schema from the signature, so with no validators the parameter is
        removed (as `value` is) rather than left for the model to guess at.
        """
        if env.secret_validators:
            fn.__doc__ += (
                f"\n\n        Pass `validator` (one of {sorted(env.secret_validators)}) to have the value checked "
                f"before it is saved; use the one that matches what the secret is for."
            )
        else:
            sig = inspect.signature(fn)
            setattr(
                fn, "__signature__", sig.replace(parameters=[p for n, p in sig.parameters.items() if n != "validator"])
            )
        return fn

    return {
        "get_secret": get_secret,
        "create_secret": _with_validator_param(
            create_secret_inline if env.allow_inline_secret_values else create_secret
        ),
        "update_secret": _with_validator_param(
            update_secret_inline if env.allow_inline_secret_values else update_secret
        ),
        "generate_secret": generate_secret,
    }
