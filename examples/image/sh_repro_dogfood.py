"""Minimal repro for the `sh: executable file not found in $PATH` remote-build failure.

from_debian_base resolves to python:3.12-slim-bookworm and layers an env var
PATH=/opt/venv/bin:$PATH. The remote builder emits each build command as
`sh -c "<cmd>"`. If that PATH is left literal (unexpanded $PATH), /bin and
/usr/bin drop off PATH and `sh` can't be found.

This image has NO Python dependencies -- just a single `with_commands`
build step -- so it isolates whether *any* sh -c step fails, independent of
uv/pip. We only build the image; we never run the task.
"""

import flyte
from flyte import Image

image = Image.from_debian_base(python_version=(3, 12), install_flyte=False).with_commands(["echo hello"])

env = flyte.TaskEnvironment(name="sh_repro", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/builder.remote.dogfood.staging.yaml")
    result = flyte.build(image, force=True, wait=True)
    print(f"URI: {result.uri}")
    print(f"Remote run: {result.remote_run}")
