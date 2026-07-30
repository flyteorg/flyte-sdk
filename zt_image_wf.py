"""End-to-end workflow for eng26-971: exercises RemoteImageChecker via a custom image.

The custom image forces the remote builder; the builder calls image_exists,
which now routes GetImage through ClusterAwareImageService (SelectCluster with
OPERATION_GET_IMAGE).

    LOG_LEVEL=debug python zt_image_wf.py

Run twice:
  - first run misses ("Image ... was not found or has expired"), builds via the
    build-image task in system/production
  - second run should hit: the "Image <name> found in remote registry" debug
    line coming after "Created ImageService client for cluster endpoint:
    https://<cluster>.dp..." is the end-to-end confirmation the lookup went
    DP-direct.

Caveat: image_exists swallows all errors into "not found", so infra failures
show up as a rebuild instead of an error; use probe_getimage.py to make
failures visible.

Not for commit; local validation only.
"""

import flyte

env = flyte.TaskEnvironment(
    name="getimage-zt-test",
    resources=flyte.Resources(memory="250Mi"),
    image=flyte.Image.from_debian_base().with_pip_packages("requests"),
)


@env.task
async def hello(name: str = "zero-trust") -> str:
    import requests  # proves the built image has the package

    return f"hello {name}, requests {requests.__version__}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(hello)
    print(run.name, run.url)
