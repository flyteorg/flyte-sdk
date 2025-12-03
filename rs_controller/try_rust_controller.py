#!/usr/bin/env python3
"""
Test script for Rust controller auth functionality.

This script tests both unary (list_tasks) and streaming (watch) gRPC calls
with authentication and retry logic.

Usage:
    export CLIENT_SECRET="your-secret-here"
    python try_rust_controller.py
"""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_list_tasks():
    """Test unary gRPC call with auth (list tasks endpoint)"""
    logger.info("=" * 60)
    logger.info("Testing unary gRPC call: list_tasks")
    logger.info("=" * 60)

    try:
        from flyte_controller_base import BaseController

        # Run the async test
        result = asyncio.run(BaseController.try_list_tasks())

        if result:
            logger.info("✅ list_tasks test PASSED")
        else:
            logger.warning("⚠️ list_tasks test returned False")

        return result

    except Exception as e:
        logger.error(f"❌ list_tasks test FAILED: {e}", exc_info=True)
        return False


def test_watch():
    """Test streaming gRPC call with auth (watch endpoint)"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing streaming gRPC call: watch")
    logger.info("=" * 60)

    try:
        from flyte_controller_base import BaseController

        # Run the async test
        result = asyncio.run(BaseController.try_watch())

        if result:
            logger.info("✅ watch test PASSED")
        else:
            logger.warning("⚠️ watch test returned False")

        return result

    except Exception as e:
        logger.error(f"❌ watch test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all tests"""
    import os

    logger.info("Starting Rust controller authentication tests")
    logger.info(f"CLIENT_SECRET env var set: {'Yes' if os.getenv('CLIENT_SECRET') else 'No (will use empty string)'}")

    results = []

    # Test 1: Unary call (list tasks)
    results.append(("list_tasks (unary)", test_list_tasks()))

    # Test 2: Streaming call (watch)
    results.append(("watch (streaming)", test_watch()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    # Exit code
    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n💥 Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    test_list_tasks()

