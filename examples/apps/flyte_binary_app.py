"""A basic app that uses the built-in Streamlit `hello` app."""

from pathlib import Path

import flyte
import flyte.app

# Built via a custom Dockerfile because flyte-demo:nightly is a k3s rootfs:
# no /usr/bin, static busybox shells, and only a partial glibc bundled for
# Postgres. The Dockerfile pulls python-build-standalone, points it at the
# bundled glibc, and installs flyte.
image = flyte.Image.from_dockerfile(
    file=Path(__file__).parent / "Dockerfile.flyte_binary",
    registry="ghcr.io/flyteorg",
    name="flyte-demo",
).clone(extendable=True).with_local_v2()

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# In this case we do not need to supply any app code
# as we are using the built-in Streamlit `hello` app.
app_env = flyte.app.AppEnvironment(
    name="flyte-oss-binary",
    image=image,
    port=30080,
    # args="streamlit hello --server.port 8080",
    resources=flyte.Resources(cpu="8", memory="8Gi"),
)


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(app_env)
    print(d[0])
