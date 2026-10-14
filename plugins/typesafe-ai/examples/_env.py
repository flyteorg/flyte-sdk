"""Shared environment for the examples.

Until `flyteplugins-typesafe-ai` is published, set `TYPESAFE_LOCAL_WHEELS=1` to
bake the wheels built by `make dist && make dist-plugins` into the image instead
of installing from PyPI. Everything else about the examples is what you would
write in your own repo.
"""

import os

import flyte

_image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("typesafe-sdk")

if os.environ.get("TYPESAFE_LOCAL_WHEELS") == "1":
    _image = _image.with_local_v2().with_local_v2_plugins("flyteplugins-typesafe-ai")
else:
    _image = _image.with_pip_packages("flyteplugins-typesafe-ai")

# The secret is mounted as TYPESAFE_API_KEY, which is the env var the SDK reads.
env = flyte.TaskEnvironment(
    name="typesafe-ai",
    image=_image,
    secrets=[flyte.Secret(key="TYPESAFE_API_KEY", as_env_var="TYPESAFE_API_KEY")],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)
