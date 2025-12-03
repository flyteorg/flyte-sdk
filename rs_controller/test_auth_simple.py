#!/usr/bin/env python3
"""
Simple test script for Rust controller auth functionality.

Usage:
    export CLIENT_SECRET="your-secret-here"
    python test_auth_simple.py
"""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_list_tasks():
    """Test unary gRPC call with auth (list tasks endpoint)"""
    from flyte_controller_base import BaseController

    logger.info("=" * 60)
    logger.info("Testing unary gRPC call: list_tasks")
    logger.info("=" * 60)

    try:
        logger.info("About to call try_list_tasks...")
        result = await BaseController.try_list_tasks()
        logger.info(f"Got result: {result}")

        if result:
            logger.info("✅ list_tasks test PASSED")
        else:
            logger.warning("⚠️ list_tasks test returned False")

        return result
    except Exception as e:
        logger.error(f"❌ list_tasks test FAILED: {e}", exc_info=True)
        return False


async def test_watch():
    """Test streaming gRPC call with auth (watch endpoint)"""
    from flyte_controller_base import BaseController

    logger.info("\n" + "=" * 60)
    logger.info("Testing streaming gRPC call: watch")
    logger.info("=" * 60)

    try:
        result = await BaseController.try_watch()

        if result:
            logger.info("✅ watch test PASSED")
        else:
            logger.warning("⚠️ watch test returned False")

        return result
    except Exception as e:
        logger.error(f"❌ watch test FAILED: {e}", exc_info=True)
        return False


async def main():
    """Run all tests"""
    import os

    logger.info("Starting Rust controller authentication tests")
    logger.info(f"CLIENT_SECRET set: {'Yes' if os.getenv('CLIENT_SECRET') else 'No (will use empty string)'}")

    # Test 1: Unary call (list tasks)
    # result1 = await test_list_tasks()
    # print(result1)

    # # Test 2: Streaming call (watch)
    result2 = await test_watch()
    print(result2)

    # # Summary
    # logger.info("\n" + "=" * 60)
    # logger.info("Test Summary")
    # logger.info("=" * 60)
    # logger.info(f"list_tasks (unary): {'✅ PASSED' if result1 else '❌ FAILED'}")
    # logger.info(f"watch (streaming): {'✅ PASSED' if result2 else '❌ FAILED'}")
    #
    # # Exit code
    # if result1 and result2:
    #     logger.info("\n🎉 All tests passed!")
    #     sys.exit(0)
    # else:
    #     logger.error("\n💥 Some tests failed")
    #     sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
