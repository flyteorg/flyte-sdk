import asyncio
import time
from typing import List, Tuple

import httpx

import flyte

env = flyte.TaskEnvironment(name="network_gremlin")


@env.task
async def sleeper(sleep: float) -> str:
    """
    A task that performs a simple sleep operation.
    """
    print(f"Sleeping for {sleep} seconds...")
    await asyncio.sleep(sleep)
    return f"Slept for {sleep} seconds"


async def network_hog() -> int:
    """
    A network hog function that performs intensive network operations,
    making multiple concurrent HTTP requests to stress network resources.
    """
    print("Starting network hog...")
    start_time = time.time()

    # List of test endpoints that can handle load
    test_urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/json",
        "https://httpbin.org/uuid",
        "https://httpbin.org/ip",
        "https://httpbin.org/user-agent",
    ]

    total_requests = 0
    successful_requests = 0

    # Create multiple batches of concurrent requests
    batch_size = 20
    num_batches = 10

    async with httpx.AsyncClient(
        timeout=30.0, limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    ) as client:
        for batch in range(num_batches):
            print(f"Starting batch {batch + 1}/{num_batches}...")

            # Create batch of concurrent requests
            tasks = []
            for i in range(batch_size):
                url = test_urls[i % len(test_urls)]
                task = asyncio.create_task(make_request(client, url, f"batch-{batch}-req-{i}"))
                tasks.append(task)
                total_requests += 1

            # Wait for all requests in this batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful requests
            for result in results:
                if not isinstance(result, Exception):
                    successful_requests += 1

            # Small delay between batches to prevent overwhelming the server
            await asyncio.sleep(0.5)

    end_time = time.time()
    duration = end_time - start_time

    print("Network hog completed!")
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Requests per second: {total_requests / duration:.2f}")

    return successful_requests


async def make_request(client: httpx.AsyncClient, url: str, request_id: str) -> dict:
    """
    Make a single HTTP request and return the response data.
    """
    try:
        response = await client.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Request {request_id} completed successfully")
            return {"status": "success", "url": url, "data": data}
        else:
            print(f"Request {request_id} failed with status {response.status_code}")
            return {"status": "failed", "url": url, "status_code": response.status_code}
    except Exception as e:
        print(f"Request {request_id} failed with exception: {e}")
        raise


@env.task
async def main(sleep: float = 5.0, n: int = 10) -> Tuple[List[str], int]:
    """
    A task that fans out to multiple instances of sleeper while running network stress test.
    """
    # Run network hog in parallel with sleeper tasks
    network_hog_task = asyncio.create_task(network_hog())
    print("Network hog task started...", flush=True)
    results = []
    for i in range(n):
        results.append(asyncio.create_task(sleeper(sleep=sleep)))

    print(f"Launching {n} sleeper tasks with {sleep} seconds each...", flush=True)
    await asyncio.sleep(0.2)  # Allow some time for tasks to start

    v = await asyncio.gather(*results)
    print("All sleeper tasks completed.", flush=True)

    successful_requests = await network_hog_task
    return v, successful_requests


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main, sleep=2.0, n=20)  # Adjust parameters as needed
    print(r.url)
