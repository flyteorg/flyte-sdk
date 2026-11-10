"""One-time web form for entering secret values without routing them through the model.

`create_secret` / `update_secret` mint a single-use token and hand the model a URL; the human
opens it in a browser and types the value into a password field. The value travels
browser -> this server -> Flyte control plane and never appears in the MCP conversation.

The token store is in memory, so it is per process: with more than one replica, the form URL
must be opened on the replica that minted it. Run one replica for a server that exposes the
secret tools (or serve them from a dedicated single-replica server).

An owner-registered validator (see `SecretValidator`) can check a value before it is saved, so a
bad token is rejected at the form instead of being found hours later by a failing task.

Form-mode elicitation is not used: the MCP specification forbids requesting sensitive values
through it, so the value is entered only in the URL-mode form.
"""

from __future__ import annotations

import asyncio
import html
import logging
import secrets
import shlex
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal
from urllib.parse import parse_qs

import flyte.remote

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    from flyte.ai.mcp._flyte_mcp_app import FlyteMCPAppEnvironment

logger = logging.getLogger(__name__)

SECRET_INPUT_PATH = "/secret-input"

#: How long a minted URL stays usable.
TOKEN_TTL_S = 10 * 60

#: Reject larger form bodies before parsing. Kubernetes caps a Secret at 1 MiB; leave room for
#: form-encoding overhead on a value near that size.
MAX_FORM_BYTES = 2 * 1024 * 1024

#: Longest a validator may run. The human is waiting on a form submit, and a probe against an
#: unreachable host would otherwise hang it; a timeout counts as a failed validation.
VALIDATOR_TIMEOUT_S = 15

SecretOp = Literal["create", "update"]

#: Checks a secret value before it is saved: returns an error sentence, or `None` when the value
#: is valid. Registered by the server's owner on `FlyteMCPAppEnvironment(secret_validators=...)`.
SecretValidator = Callable[[str], Awaitable["str | None"]]

# No scripts, no external assets, and the form can only post back to this origin.
_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"
    ),
}


async def _secret_exists(name: str) -> bool:
    """Whether the secret `name` exists. Call inside `env._scoped(...)`, which sets the project/domain.

    Only NOT_FOUND means "no"; any other error (permissions, an unreachable control plane) says
    nothing about existence and propagates.
    """
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    try:
        await flyte.remote.Secret.get.aio(name)  # type: ignore[attr-defined]
    except ConnectError as e:
        if e.code != Code.NOT_FOUND:
            raise
        return False
    return True


def _already_exists_message(name: str) -> str:
    return f"Secret '{name}' already exists, so nothing was changed. Use `update_secret` to replace its value."


def _missing_message(name: str) -> str:
    return f"Secret '{name}' does not exist, so nothing was changed. Use `create_secret` to create it."


def _cli_fallback(op: SecretOp, name: str, project: str, domain: str) -> str:
    # The command is meant to be pasted into a shell, and the name comes from the model.
    name, project, domain = shlex.quote(name), shlex.quote(project), shlex.quote(domain)
    create = f"flyte create secret {name} -p {project} -d {domain}"
    if op == "create":
        return create
    # There is no `flyte update secret`; replacing a value means deleting and re-creating it.
    return f"flyte delete secret {name} -p {project} -d {domain} && {create}"


def _http_request(ctx: Any) -> Request | None:
    """The Starlette request behind a tool call, or `None` when there isn't one."""
    try:
        return ctx.request_context.request if ctx is not None else None
    except ValueError:  # FastMCP's Context raises this outside of a request
        return None


def _public_base_url(request: Request) -> str:
    """Scheme + host + any proxy prefix of the *outer* app, which serves the form.

    `/secret-input` is routed on the outer Starlette app, so neither the MCP mount nor anything
    the transport nests below it (SSE's `/messages`) belongs in the URL.
    """
    scope = request.scope
    # Starlette records the root_path seen before the first Mount (i.e. the proxy prefix only) as
    # app_root_path and carries it through nested mounts; FastMCP's SSE transport nests one.
    root_path = scope.get("app_root_path")
    if root_path is None:  # not routed through a Mount at all
        root_path = scope.get("root_path", "")
    return f"{request.url.scheme}://{request.url.netloc}{root_path.rstrip('/')}"


def _client_supports_url_elicitation(ctx: Any) -> bool:
    """Whether the client advertised URL-mode elicitation.

    Feature-detected attribute by attribute: URL elicitation arrived in mcp 1.26 (the minimum
    pin), but a client that predates it simply won't advertise the `url` capability.
    """
    try:
        elicitation = ctx.session.client_params.capabilities.elicitation
    except (AttributeError, ValueError):
        return False
    return getattr(elicitation, "url", None) is not None


def _now() -> float:
    return time.time()


@dataclass
class PendingSecret:
    op: SecretOp
    name: str
    project: str
    domain: str
    expires_at: float
    used: bool = False
    #: A POST is being handled. Separate from `used`: `_purge()` drops used tokens, and a request
    #: that is rejected without burning the token must hand it back intact.
    in_flight: bool = False
    validator: str | None = None


class SecretInputStore:
    """In-memory map of one-time tokens to the secret write they authorize."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingSecret] = {}

    def mint(
        self, op: SecretOp, name: str, project: str, domain: str, validator: str | None = None
    ) -> tuple[str, PendingSecret]:
        self._purge()
        token = secrets.token_urlsafe(32)
        pending = PendingSecret(op, name, project, domain, expires_at=_now() + TOKEN_TTL_S, validator=validator)
        self._pending[token] = pending
        return token, pending

    def get(self, token: str) -> PendingSecret | None:
        return self._pending.get(token)

    def _purge(self) -> None:
        # Minting is the only thing that grows the store, so sweeping here bounds it without a timer.
        now = _now()
        for token in [t for t, p in self._pending.items() if p.used or p.expires_at <= now]:
            del self._pending[token]


async def run_validator(validator: SecretValidator, value: str) -> str | None:
    """Run `validator` on `value`; an error sentence if it is not valid, else `None`.

    A validator that raises or times out is a failed validation, not a server error. The message
    names only the exception class: `str(e)` could quote the value.
    """
    try:
        message = await asyncio.wait_for(validator(value), VALIDATOR_TIMEOUT_S)
    except asyncio.TimeoutError:
        return f"The value could not be checked: the validator did not answer within {VALIDATOR_TIMEOUT_S} s."
    except Exception as e:
        logger.warning("secret validator failed: %s", type(e).__name__)
        return f"The value could not be checked: the validator failed ({type(e).__name__})."
    # A validator is owner code, but one that echoes its input would put the value on the page
    # or in front of the model.
    return message.replace(value, "[redacted]") if message else None


def _page(title: str, body: str, status: int = 200) -> Response:
    from starlette.responses import HTMLResponse

    doc = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:34rem;margin:3rem auto;padding:0 1rem}"
        "dl{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1rem}dd{margin:0;font-family:monospace}"
        "input[type=password],textarea{width:100%;box-sizing:border-box;margin:.5rem 0;font:inherit}"
        "button{font:inherit;padding:.4rem 1rem}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
    )
    return HTMLResponse(doc, status_code=status, headers=_HEADERS)


def _summary(p: PendingSecret) -> str:
    expires = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(p.expires_at))
    rows = [("Action", p.op), ("Secret", p.name), ("Project", p.project), ("Domain", p.domain), ("Expires", expires)]
    return "<dl>" + "".join(f"<dt>{k}</dt><dd>{html.escape(v)}</dd>" for k, v in rows) + "</dl>"


def _form(p: PendingSecret) -> Response:
    # A <details> textarea instead of a script-driven toggle keeps the page script-free (see CSP).
    body = (
        f"{_summary(p)}"
        "<form method=post>"
        "<label>Value<input type=password name=value autocomplete=off autofocus></label>"
        "<details><summary>Multi-line value (certificates, keys)</summary>"
        "<textarea name=value_multiline rows=8 autocomplete=off spellcheck=false></textarea></details>"
        f"<button type=submit>{html.escape(p.op.capitalize())} secret</button></form>"
        "<p>The value is sent straight to Flyte and is not shown to the assistant. "
        "This link works once.</p>"
    )
    return _page(f"{p.op.capitalize()} secret", body)


def _rejection(p: PendingSecret | None) -> Response | None:
    """The 404/410 page for a token that cannot be used, or `None` if it can."""
    if p is None:
        return _page("Link not found", "<p>This link is unknown. Ask the assistant for a new one.</p>", 404)
    if p.used or p.in_flight or p.expires_at <= _now():
        reason = "already been used" if p.used or p.in_flight else "expired"
        return _page("Link no longer valid", f"<p>This link has {reason}. Ask the assistant for a new one.</p>", 410)
    return None


def _extract_value(raw_body: bytes) -> str | None:
    """Pull the submitted value out of a form-encoded body; `None` if empty or ambiguous."""
    # Parsed by hand: Starlette's Request.form() needs python-multipart even for urlencoded bodies.
    fields = parse_qs(raw_body.decode("utf-8", errors="replace"), keep_blank_values=True)
    single = (fields.get("value") or [""])[0]
    # Browsers submit textarea newlines as CRLF; a PEM block with stray \r is not what the user typed.
    multi = (fields.get("value_multiline") or [""])[0].replace("\r\n", "\n")
    if bool(single) == bool(multi):
        return None
    return single or multi


def secret_input_routes(env: FlyteMCPAppEnvironment) -> list[Route]:
    """The GET/POST `/secret-input/{token}` routes, bound to `env`'s token store and scoping."""
    from starlette.routing import Route

    store = env._secret_inputs

    async def show_form(request: Request) -> Response:
        pending = store.get(request.path_params["token"])
        rejected = _rejection(pending)
        if rejected is not None:
            return rejected
        assert pending is not None
        return _form(pending)

    async def submit(request: Request) -> Response:
        pending = store.get(request.path_params["token"])
        rejected = _rejection(pending)
        if rejected is not None:
            return rejected
        assert pending is not None

        # Claim the token before the first await. Burning it (below) happens only after the body,
        # the validator and the existence check have awaited, so without a claim two POSTs of one
        # link would both get that far and both write.
        pending.in_flight = True
        try:
            return await _handle_post(request, pending)
        finally:
            # Hand the link back unless it was burned: every "this link is still valid" page
            # below promises a retry.
            if not pending.used:
                pending.in_flight = False

    async def _handle_post(request: Request, pending: PendingSecret) -> Response:
        if int(request.headers.get("content-length") or 0) > MAX_FORM_BYTES:
            return _page("Value too large", "<p>The value is larger than 2 MiB.</p>", 413)
        # Counted as it arrives: a chunked POST has no Content-Length to check up front.
        chunks: list[bytes] = []
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > MAX_FORM_BYTES:
                return _page("Value too large", "<p>The value is larger than 2 MiB.</p>", 413)
            chunks.append(chunk)
        raw = b"".join(chunks)
        value = _extract_value(raw)
        if value is None:
            return _page(
                "Enter a value",
                "<p>Fill in exactly one of the two fields. Go back and try again; this link is still valid.</p>",
                400,
            )

        if pending.validator is not None:
            assert env.secret_validators is not None  # a token only names a validator the env registered
            rejection = await run_validator(env.secret_validators[pending.validator], value)
            if rejection is not None:
                # Not burned: the human can retype and resubmit.
                return _page(
                    "Value rejected",
                    f"<p>{html.escape(rejection)}</p><p>Go back and try again; this link is still valid.</p>",
                    422,
                )

        # The secret may have been created or deleted since the URL was minted, and the control
        # plane silently overwrites on create and creates on update, so check again right before
        # writing. Not burned on a mismatch: the link can never succeed, but the human learns why.
        try:
            with env._scoped(pending.project, pending.domain):
                exists = await _secret_exists(pending.name)
        except Exception as e:
            logger.warning("secret existence check of %r failed: %s", pending.name, type(e).__name__)
            message = str(e).replace(value, "[redacted]")
            return _page(
                "Could not check the secret",
                f"<p><code>{html.escape(type(e).__name__)}</code>: {html.escape(message)}</p>"
                "<p>Nothing was changed. Go back and try again; this link is still valid.</p>",
                502,
            )
        if exists != (pending.op == "update"):
            what = "already exists" if exists else "does not exist"
            return _page(
                "Secret not changed",
                f"<p>Secret <code>{html.escape(pending.name)}</code> {what} in "
                f"{html.escape(pending.project)}/{html.escape(pending.domain)}, so nothing was changed. "
                "This link can no longer succeed. Ask the assistant for a new link.</p>",
                409,
            )

        # Burn the token before the SDK call: if the call is slow or fails half-way, a retry
        # must come from a fresh, deliberately requested URL rather than a replayed form.
        pending.used = True
        try:
            with env._scoped(pending.project, pending.domain):
                writer: Any = flyte.remote.Secret.create if pending.op == "create" else flyte.remote.Secret.update
                await writer.aio(name=pending.name, value=value)
        except Exception as e:
            # Log the class only: an exception message could in principle quote the request.
            logger.warning("secret %s of %r failed: %s", pending.op, pending.name, type(e).__name__)
            message = str(e).replace(value, "[redacted]")
            return _page(
                "Could not save the secret",
                f"<p><code>{html.escape(type(e).__name__)}</code>: {html.escape(message)}</p>"
                "<p>This link is now spent. Ask the assistant for a new one to retry.</p>",
                502,
            )
        done = "created" if pending.op == "create" else "updated"
        return _page(
            f"Secret {done}",
            f"<p>Secret <code>{html.escape(pending.name)}</code> was {done} in "
            f"{html.escape(pending.project)}/{html.escape(pending.domain)}. You can close this tab.</p>",
        )

    async def handle(request: Request) -> Response:
        return await (submit if request.method == "POST" else show_form)(request)

    return [Route(f"{SECRET_INPUT_PATH}/{{token}}", endpoint=handle, methods=["GET", "POST"])]
