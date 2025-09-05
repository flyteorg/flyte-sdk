from __future__ import annotations

import asyncio

import flyte
from flyte._internal.controllers.remote._client import ControllerClient
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyteidl2.common import identifier_pb2
from flyteidl2.workflow import (
    queue_service_pb2,
    task_definition_pb2,
)
from flyte.models import SerializationContext

env = flyte.TaskEnvironment(name="hello_world")


@env.task
async def say_hello():
    print("Hello World")


async def get_local_client() -> ControllerClient:
    endpoint = "localhost:8090"
    client = await ControllerClient.for_endpoint(endpoint=endpoint, insecure=True)
    return client


async def enqueue_run(client: ControllerClient):
    run_id = identifier_pb2.RunIdentifier(
        org="testorg",
        project="project",
        domain="development",
        name="root_run",
    )

    sc = SerializationContext(
        code_bundle=None,
        version="abc123",
        input_path="s3://bucket/test/run/inputs.pb",
        output_path="s3://bucket/outputs/0/jfkljfa/0",
    )

    task_spec = translate_task_to_wire(say_hello, sc)

    enqueue_request = queue_service_pb2.EnqueueActionRequest(
        action_id=identifier_pb2.ActionIdentifier(
            name="subrun-1",
            run=run_id,
        ),
        parent_action_name="root_run",
        task=queue_service_pb2.TaskAction(
            id=task_definition_pb2.TaskIdentifier(
                org="testorg",
                project="project",
                domain="development",
                name="say_hello",
                version="abc123",
            ),
            spec=task_spec,
        ),
        input_uri="s3://bucket/test/run/inputs.pb",
        output_uri="s3://bucket/outputs/0/jfkljfa/0",
        group="",
    )

    await client.queue_service.EnqueueAction(enqueue_request)


async def main():
    client = await get_local_client()
    await enqueue_run(client)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
