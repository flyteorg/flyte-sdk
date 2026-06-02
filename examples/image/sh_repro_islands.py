"""Same minimal image as sh_repro_dogfood.py, but built against islands.

If this fails with `sh: executable file not found in $PATH` while the identical
definition succeeds on dogfood, the cause is environmental (islands's
base-image config resolve), not the image definition.
"""

import flyte
from flyte import Image

image = Image.from_debian_base(python_version=(3, 12), install_flyte=False).with_commands(["echo hello"])

env = flyte.TaskEnvironment(name="sh_repro", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config(
        "/Users/ytong/go/src/github.com/unionai/cloud/gen/cli-config/uctl/islands.production_v2.yaml"
    )
    result = flyte.build(image, force=True, wait=True)
    print(f"URI: {result.uri}")
    print(f"Remote run: {result.remote_run}")
