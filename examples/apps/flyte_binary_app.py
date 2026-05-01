"""A basic app that uses the built-in Streamlit `hello` app."""

from pathlib import Path

from kubernetes.client.models import (
    V1Capabilities,
    V1Container,
    V1PodSpec,
    V1SecurityContext,
)

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
)

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# In this case we do not need to supply any app code
# as we are using the built-in Streamlit `hello` app.
# k3s-in-a-pod needs to write /sys/fs/cgroup, create netns, run iptables, etc.
# Ask for a privileged container + SYS_ADMIN. The primary container name must
# be "app" (see src/flyte/app/_runtime/app_serde.py:110); the SDK merges the
# runtime-generated image/command/resources into this container, so we only
# set the security bits here. If the Union cluster's admission policy forbids
# privileged pods, the deploy will fail with a clear admission error.
privileged_pod_template = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="app",
                security_context=V1SecurityContext(
                    privileged=True,
                    allow_privilege_escalation=True,
                    run_as_user=0,
                    capabilities=V1Capabilities(add=["SYS_ADMIN", "NET_ADMIN"]),
                ),
            ),
        ],
    ),
)

app_env = flyte.app.AppEnvironment(
    name="flyte-oss-binary",
    image=image,
    port=30080,
    # Run the same entrypoint the base image ships with (k3d wrapper that
    # boots k3s + flyte-demo-bootstrap).
    command=["/bin/k3d-entrypoint.sh", "server", "--disable=servicelb", "--disable=metrics-server"],
    resources=flyte.Resources(cpu="8", memory="8Gi"),
    pod_template=privileged_pod_template,
)


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(app_env)
    print(d[0])
