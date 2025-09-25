import asyncio

from flyte._internal.controllers.remote._client import ControllerClient
from flyte._protos.common import identifier_pb2
from flyte._protos.workflow import state_service_pb2


async def watch_direct():
    """
    Test the watch functionality directly without informer wrapper.
    """

    # ==================== TEST CONFIGURATION ====================
    # Fill in these values to test with real data:

    RUN_NAME = "r9ssgff995xrpx9bjq7m"  # Replace with actual run name
    PARENT_ACTION_NAME = "a0"  # Replace with actual parent action name

    # Connection configuration
    ENDPOINT = "dns:///dogfood.cloud-staging.union.ai"
    INSECURE = False

    # ============================================================

    print("\n🧪 Starting Direct Watch Test")
    print(f"🌐 Endpoint: {ENDPOINT}")
    print(f"🔧 Run Name: {RUN_NAME}")
    print(f"🔧 Parent Action: {PARENT_ACTION_NAME}")
    print("=" * 60)

    # Create controller client using the proper method
    controller_client = await ControllerClient.for_endpoint(
        endpoint=ENDPOINT,
        auth_type="Pkce",
    )

    # Setup run identifier
    run_id = identifier_pb2.RunIdentifier(project="flytesnacks", domain="development", org="dogfood", name=RUN_NAME)

    print("\n🚀 Creating watcher...")

    try:
        # This is the exact code you want to test from _informer.py
        watcher = controller_client.state_service.Watch(
            state_service_pb2.WatchRequest(
                parent_action_id=identifier_pb2.ActionIdentifier(
                    name=PARENT_ACTION_NAME,
                    run=run_id,
                ),
            ),
            wait_for_ready=True,
        )

        print("✅ Watcher created successfully")
        print("⏳ Starting to iterate over responses...")

        count = 0
        async for resp in watcher:
            count += 1
            print(f"\n📦 Response #{count}:", flush=True)
            print(f"   Raw response: {resp}", flush=True)

        #     if resp.control_message is not None and resp.control_message.sentinel:
        #         print(f"   🚩 Control message - Sentinel: {resp.control_message.sentinel}")
        #
        #     if resp.action_update:
        #         action = resp.action_update
        #         print(f"   🎯 Action: {action.action_id.name}")
        #         print(f"   📊 Phase: {run_definition_pb2.Phase.Name(action.phase)}")
        #         if action.output_uri:
        #             print(f"   📁 Output URI: {action.output_uri}")
        #         if action.HasField('error'):
        #             print(f"   ❌ Error: {action.error}")
        #
        #     # Limit to avoid infinite loop for testing
        #     if count >= 20:
        #         print(f"\n⏹️  Stopping after {count} responses for testing")
        #         break
        #
        # if count == 0:
        #     print("   📭 No responses received. Check if:")
        #     print("      - RUN_NAME and PARENT_ACTION_NAME are correct")
        #     print("      - The run is active and has sub-actions")
        #     print("      - Authentication is working properly")

    except Exception as e:
        print(f"❌ Error during watch: {e}")
        print("🔧 Make sure to fill in the correct RUN_NAME and PARENT_ACTION_NAME")
        raise

    finally:
        print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(watch_direct())
