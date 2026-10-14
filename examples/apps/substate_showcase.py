"""Deploy one app per failure mode, so every app status/substate badge in the
console's Apps listing can be verified against a real cluster.

Each app below is deliberately broken in a specific way. Deploy them all, then
open the Apps page and check the badge each one renders.

    python examples/apps/substate_showcase.py            # deploy all
    python examples/apps/substate_showcase.py --cleanup  # delete all

Every app pins ``replicas=(1, 1)`` rather than the default scale-to-zero, so a
broken pod keeps retrying and its badge stays visible instead of idling away.
Because each app sets ``command``, the container entrypoint is fully overridden
and no Flyte runtime is needed inside the image -- which is what lets the two
image-related cases below use raw image URIs.

Expected badges (substates come from the backend, so timing varies -- the
error states typically take 30-90s to surface after deploy):

    ok         Active                          healthy control, shows as a card
    crashloop  Failed - CrashLoopBackOff       container exits 1 immediately
    imagepull  Failed - ImagePullError         image URI does not exist
    oom        Failed - OOM Killed             allocates past a 256Mi limit
    secret     Failed - Secret mount error     references a nonexistent secret
    init       Pending - Initializing          runs, but never binds the port
    bigimage   Pending - Pulling Image         ~3GB pull, then goes Active

Not covered: Webhook error. That substate comes from a Kubernetes admission
webhook rejecting the pod, which can't be provoked from the SDK.

Note on images: the two image-related cases pass a plain string URI instead of
a ``flyte.Image``. A ``flyte.Image`` is sent to the image builder first, so a
nonexistent base would fail the build locally and the app would never reach the
cluster -- no ImagePullError badge to look at. A string URI skips the builder
and lands in the pod spec as-is, so the pull failure happens where we want it,
on the cluster.
"""

import asyncio
import sys

import flyte
import flyte.remote
from flyte.app import AppEnvironment, Scaling

# flyte.serve() ends by watching the app until it reports "activated", and most
# of the apps here are built never to get there. So each serve is given a
# deadline and then abandoned. The app itself is created on the cluster well
# before that watch begins, so giving up on the watch leaves the app deployed
# and failing where we can see it -- which is the whole point.
DEPLOY_TIMEOUT_S = 90

# Shared by the cases whose failure is behavioral rather than image-related.
# One image, built once, reused by all of them.
PY_IMAGE = flyte.Image.from_debian_base(python_version=(3, 12))

ALWAYS_ON = Scaling(replicas=(1, 1))
SMALL = flyte.Resources(cpu=1, memory="512Mi")

app_envs = [
    # Active -- healthy control. Renders as a card in the active section.
    AppEnvironment(
        name="substate-ok",
        image=PY_IMAGE,
        command="python -m http.server 8080",
        port=8080,
        resources=SMALL,
        scaling=ALWAYS_ON,
        requires_auth=False,
    ),
    # CrashLoopBackOff -- exits nonzero on every restart.
    AppEnvironment(
        name="substate-crashloop",
        image=PY_IMAGE,
        command=["sh", "-c", "echo 'crashing on purpose'; exit 1"],
        port=8080,
        resources=SMALL,
        scaling=ALWAYS_ON,
        requires_auth=False,
    ),
    # OOM Killed -- 50MB at a time against a 256Mi limit.
    AppEnvironment(
        name="substate-oom",
        image=PY_IMAGE,
        command=[
            "python",
            "-c",
            "blocks = []\nwhile True:\n    blocks.append(bytearray(50 * 1024 * 1024))",
        ],
        port=8080,
        resources=flyte.Resources(cpu=1, memory="256Mi"),
        scaling=ALWAYS_ON,
        requires_auth=False,
    ),
    # Secret mount error -- the secret is never created on the cluster.
    AppEnvironment(
        name="substate-secret",
        image=PY_IMAGE,
        command="python -m http.server 8080",
        port=8080,
        resources=SMALL,
        scaling=ALWAYS_ON,
        requires_auth=False,
        secrets="substate-showcase-missing-secret",
    ),
    # Initializing -- process stays up but nothing ever listens on the port,
    # so the readiness probe never passes.
    AppEnvironment(
        name="substate-init",
        image=PY_IMAGE,
        command=["sh", "-c", "sleep 3600"],
        port=8080,
        resources=SMALL,
        scaling=ALWAYS_ON,
        requires_auth=False,
    ),
    # ImagePullError -- string URI, see the module docstring.
    AppEnvironment(
        name="substate-imagepull",
        image="ghcr.io/unionai/substate-showcase-does-not-exist:v0",
        command="python -m http.server 8080",
        port=8080,
        resources=SMALL,
        scaling=ALWAYS_ON,
        requires_auth=False,
    ),
    # Pulling Image -- a genuinely large public image, so the pull is slow
    # enough to catch in the UI before the app settles into Active.
    AppEnvironment(
        name="substate-bigimage",
        image="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        command="python -m http.server 8080",
        port=8080,
        resources=flyte.Resources(cpu=1, memory="2Gi"),
        scaling=ALWAYS_ON,
        requires_auth=False,
    ),
]


async def _deploy_one(env: AppEnvironment):
    try:
        app = await asyncio.wait_for(flyte.serve.aio(env), timeout=DEPLOY_TIMEOUT_S)
        print(f"{env.name}: active at {app.url}")
    except asyncio.TimeoutError:
        # Expected for every deliberately-broken app: deployed, just never
        # healthy. Check the badge in the console.
        print(f"{env.name}: deployed, not active after {DEPLOY_TIMEOUT_S}s")
    except RuntimeError as e:
        # The activation watch raises once an app reports FAILED. For the
        # broken apps that's the destination, not a problem -- the app is on
        # the cluster with a badge to look at.
        print(f"{env.name}: deployed, reached FAILED ({e})")
    except Exception as e:
        # Anything else is a real deploy-time error (client or control plane)
        # and means no app was created at all.
        print(f"{env.name}: DEPLOY FAILED: {e}")


async def _deploy_all():
    # Build up front, once. The serves below run concurrently, and without
    # this the five apps sharing PY_IMAGE would each kick off their own build.
    await flyte.build_images.aio(*app_envs)
    await asyncio.gather(*(_deploy_one(env) for env in app_envs))


def deploy():
    asyncio.run(_deploy_all())


def cleanup():
    for env in app_envs:
        try:
            # Delete refuses anything that isn't stopped, so deactivate first.
            # The broken apps never reach a healthy state, so don't wait on them.
            try:
                flyte.remote.App.get(env.name).deactivate()
            except Exception as e:
                print(f"  ({env.name}: deactivate said {e})")
            flyte.remote.App.delete(env.name)
            print(f"deleted {env.name}")
        except Exception as e:
            print(f"FAILED to delete {env.name}: {e}")


if __name__ == "__main__":
    # --config points at a specific cluster; without it the usual discovery order
    # applies (./config.yaml, ./.flyte/config.yaml, ~/.flyte/config.yaml, ...).
    config_path = None
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]

    flyte.init_from_config(config_path)
    if "--cleanup" in sys.argv:
        cleanup()
    else:
        deploy()
