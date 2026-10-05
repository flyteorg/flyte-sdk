"""Local union-devbox only: stamp the `x-user-subject` header the devbox's app service requires.

A hosted tenant derives the caller's identity from authentication; the local devbox has no ingress identity
filter, so its app service expects the caller to send `x-user-subject` (the local console does the same). The
flyte CLI deliberately does not, so deploying an app from the CLI against a devbox fails with
"x-user-subject header not found".

`run_e2e.sh` puts this directory on PYTHONPATH when its config points at localhost; Python then imports this
file at startup and patches the CLI's metadata interceptor. Do not use it against a real deployment, and do not
copy it into the SDK. To use it by hand:

    PYTHONPATH=examples/artifacts/devbox_shim flyte deploy --root-dir . apps/scoring.py scoring
"""

try:
    from flyte.remote._client.auth._interceptors import default_metadata as _dm

    _orig_on_start = _dm.DefaultMetadataInterceptor.on_start

    async def _on_start(self, ctx):
        ctx.request_headers().setdefault("x-user-subject", "devbox-user")
        return await _orig_on_start(self, ctx)

    _dm.DefaultMetadataInterceptor.on_start = _on_start
except Exception:  # flyte not importable in this interpreter: nothing to patch
    pass
