import asyncio

# from cloud_mod.cloud_mod import cloudidl
# from cloud_mod.cloud_mod import Action
from pathlib import Path

from flyte_controller_base import Action, BaseController, cloudidl

from examples.advanced.hybrid_mode import say_hello_hybrid
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte.models import (
    CodeBundle,
    SerializationContext,
)

img_cache = ImageCache.from_transport(
    "H4sIAAAAAAAC/wXBSQ6AIAwAwL/0TsG6hs8YlILEpUbFxBj/7swLaXWR+0VkzjvYF1y+BCzEaTwwic5bks0lJeepw/JcbPenxKJUt0FCM1CLnu+KVAwjd559g54M1aYtavi+H56TcPxgAAAA"
)
s_ctx = SerializationContext(
    project="testproject",
    domain="development",
    org="testorg",
    code_bundle=CodeBundle(
        computed_version="605136feba679aeb1936677f4c5593f6",
        tgz="s3://bucket/testproject/development/MBITN7V2M6NOWGJWM57UYVMT6Y======/fast0dc2ef669a983610a0b9793e974fb288.tar.gz",
    ),
    version="605136feba679aeb1936677f4c5593f6",
    image_cache=img_cache,
    root_dir=Path("/Users/ytong/go/src/github.com/unionai/unionv2"),
)
task_spec = translate_task_to_wire(say_hello_hybrid, s_ctx)
xxx = task_spec.SerializeToString()

yyy = cloudidl.workflow.TaskSpec.decode(xxx)
print(yyy)


class MyRunner(BaseController):
    ...
    # play around with this
    # def __init__(self, run_id: cloudidl.workflow.RunIdentifier):
    #     super().__new__(BaseController, run_id)


async def main():
    run_id = cloudidl.workflow.RunIdentifier(
        org="testorg", domain="development", name="rxp79l5qjpmmdd84qg7j", project="testproject"
    )

    sub_action_id = cloudidl.workflow.ActionIdentifier(name="sub_action_3", run=run_id)

    action = Action.from_task(
        sub_action_id=sub_action_id,
        parent_action_name="a0",
        group_data=None,
        task_spec=yyy,
        inputs_uri="s3://bucket/metadata/v2/testorg/testproject/development/rllmmzgh6v4xjc8pswc8/4jzwmmj06fnpql20rtlqz4aq2/inputs.pb",
        run_output_base="s3://bucket/metadata/v2/testorg/testproject/development/rllmmzgh6v4xjc8pswc8",
        cache_key=None,
    )

    runner = MyRunner(run_id=run_id)

    result = await runner.submit_action(action)
    print("First submit done", flush=True)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
