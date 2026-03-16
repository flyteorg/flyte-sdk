"""Build and push a custom image from CI.

Define your image below, then run:

    flyte build examples/image/ci_build_image.py env
"""

import flyte

env = flyte.Environment(
    name="env",
    image=flyte.Image.from_debian_base(
        registry="ghcr.io/myorg",
        name="myimage",
    ),
)
