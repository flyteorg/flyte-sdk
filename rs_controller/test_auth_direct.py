#!/usr/bin/env python3
"""
Direct test of auth metadata service without middleware
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_auth_service_direct():
    """Test calling auth metadata service directly"""
    from flyte_controller_base import BaseController

    # This will try to call the auth service
    logger.info("Testing direct auth service call...")
    result = await BaseController.try_list_tasks()
    logger.info(f"Result: {result}")
    return result


if __name__ == "__main__":
    asyncio.run(test_auth_service_direct())
