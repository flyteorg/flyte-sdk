"""A basic app that uses the built-in Streamlit `hello` app."""

import os

import flyte
import flyte.app

# streamlit must allow protobuf>=6 (>=1.46 relaxed the cap to <7); older pins force
# protobuf<6, which is incompatible with flyteidl2's gencode and crashes the app.
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("streamlit>=1.46")

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# In this case we do not need to supply any app code
# as we are using the built-in Streamlit `hello` app.
# Streamlit's WebSocket handshake (/_stcore/stream) goes through the `websockets`
# library, whose per-header-line cap defaults to 8 KiB (WEBSOCKETS_MAX_LINE_LENGTH)
# and header-count cap to 128 (WEBSOCKETS_MAX_NUM_HEADERS). These default to the
# library values (unchanged behavior) but are overridable via env vars: behind an
# auth proxy that injects large headers (big IdP tokens or forwarded session
# cookies) the handshake can exceed the cap and the stream never upgrades, so a
# deployment can raise them here without editing the app.
_ws_max_line = os.environ.get("WEBSOCKETS_MAX_LINE_LENGTH", str(8 * 1024))  # websockets default: 8 KiB
_ws_max_headers = os.environ.get("WEBSOCKETS_MAX_NUM_HEADERS", "128")  # websockets default: 128

app_env = flyte.app.AppEnvironment(
    name="streamlit-hello-v2",
    image=image,
    args="streamlit hello --server.port 8080",
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    env_vars={
        "WEBSOCKETS_MAX_LINE_LENGTH": _ws_max_line,
        "WEBSOCKETS_MAX_NUM_HEADERS": _ws_max_headers,
    },
)


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(app_env)
    print(d[0])
