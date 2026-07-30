"""Probe for eng26-971: validate that GetImage routes via SelectCluster.

Hits the staging cluster service's SelectCluster with OPERATION_GET_IMAGE, then
makes the actual routed GetImage call through ClusterAwareImageService.

Run with debug logging so the wrapper's routing line is visible:

    LOG_LEVEL=debug python probe_getimage.py

Expected with zero trust ON for the org:
  - SelectCluster prints a DP-direct URL like https://<cluster>.dp.<cloudHost>
  - debug log shows "Created ImageService client for cluster endpoint: ..."
  - a NOT_FOUND-style error for the fake image name is a SUCCESS: it proves the
    call reached the dataplane image service over the DP-direct transport.

Expected with zero trust OFF:
  - endpoint comes back as the control-plane host (Host-header fallback)
  - no "Created ImageService client" line; the default CP client is used.

Not for commit; local validation only.
"""

import asyncio

import flyte
from flyteidl2.cluster import payload_pb2 as cluster_pb2
from flyteidl2.common import identifier_pb2
from flyteidl2.imagebuilder import definition_pb2
from flyteidl2.imagebuilder import payload_pb2 as image_pb2

from flyte._initialize import _get_init_config


async def main():
    await flyte.init_from_config.aio()  # or: await flyte.init.aio(endpoint=..., org=..., project=..., domain=...)
    cfg = _get_init_config()

    # 1) raw SelectCluster: what endpoint does staging hand back for GET_IMAGE?
    req = cluster_pb2.SelectClusterRequest(
        operation=cluster_pb2.SelectClusterRequest.Operation.OPERATION_GET_IMAGE,
    )
    req.project_id.CopyFrom(
        identifier_pb2.ProjectIdentifier(organization=cfg.org, domain=cfg.domain, name=cfg.project)
    )
    resp = await cfg.client.cluster_service.select_cluster(req)
    print(f"SelectCluster(OPERATION_GET_IMAGE) -> {resp.cluster_endpoint!r}")

    # 2) full path through the new ClusterAwareImageService
    get_req = image_pb2.GetImageRequest(
        id=definition_pb2.ImageIdentifier(name="definitely-not-a-real-image:v1"),
        organization=cfg.org,
        project_id=identifier_pb2.ProjectIdentifier(
            organization=cfg.org, domain=cfg.domain, name=cfg.project
        ),
    )
    try:
        img = await cfg.client.image_service.get_image(get_req)
        print(f"GetImage OK: {img.image.fqin}")
    except Exception as e:
        print(f"GetImage returned: {type(e).__name__}: {e}")


asyncio.run(main())
