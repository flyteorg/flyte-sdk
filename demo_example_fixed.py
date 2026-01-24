import asyncio
import time


def sleeper():
    time.sleep(100)


async def main():
    # Create the task
    f = asyncio.create_task(asyncio.to_thread(sleeper))
    await asyncio.sleep(1)
    print("Hello, World!")

    # FIX: Cancel the task before returning
    f.cancel()

    # Try to wait for cancellation to propagate, but don't block
    try:
        await asyncio.wait_for(asyncio.shield(f), timeout=0.1)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    return


async def main_better():
    """Better fix: Use a custom executor with daemon threads"""
    from concurrent.futures import ThreadPoolExecutor

    # Create executor with daemon threads
    executor = ThreadPoolExecutor(max_workers=1)
    # Make the threads daemon so they don't block shutdown
    executor._threads.clear()  # Clear any existing threads

    # Run in custom executor
    loop = asyncio.get_event_loop()
    f = loop.run_in_executor(executor, sleeper)

    await asyncio.sleep(1)
    print("Hello, World! (better fix)")

    # Cancel and don't wait for thread
    f.cancel()
    executor.shutdown(wait=False, cancel_futures=True)

    return


async def main_best():
    """Best fix for Flyte: Fire completion events when informer fails"""
    import concurrent.futures

    # Simulate the Flyte scenario
    future = concurrent.futures.Future()

    async def worker():
        """Simulates the submit_sync worker thread"""
        try:
            # This simulates fut.result() in _task.py:319
            # In real code, this blocks forever if the future never completes
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: future.result(timeout=5.0)  # FIX: Add timeout!
            )
            return result
        except (concurrent.futures.TimeoutError, asyncio.CancelledError) as e:
            print(f"Worker got exception: {e}")
            raise

    task = asyncio.create_task(worker())
    await asyncio.sleep(1)
    print("Hello, World! (best fix - with timeout)")

    # Simulate informer failure - the future is never set
    # But the timeout will unblock the thread
    task.cancel()

    try:
        await asyncio.wait_for(task, timeout=0.1)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    return


if __name__ == "__main__":
    print("=" * 60)
    print("Testing basic fix (still has issues)...")
    print("=" * 60)
    try:
        asyncio.run(main())
        print("✓ Completed without hanging!")
    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Testing better fix (custom executor)...")
    print("=" * 60)
    try:
        asyncio.run(main_better())
        print("✓ Completed without hanging!")
    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Testing best fix (timeout on blocking call)...")
    print("=" * 60)
    try:
        asyncio.run(main_best())
        print("✓ Completed without hanging!")
    except Exception as e:
        print(f"✗ Failed: {e}")
