"""
Demonstrates the hang and the fix for the Flyte blocking thread issue.
"""
import asyncio
import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor


def sleeper():
    """Simulates a blocking call like fut.result() in Thread 33"""
    print("Thread: Starting to sleep for 100 seconds...")
    time.sleep(100)
    print("Thread: Woke up!")


# ============================================================================
# BROKEN VERSION - This will hang
# ============================================================================
async def main_broken():
    print("\n[BROKEN] Creating task that will hang on shutdown...")
    f = asyncio.create_task(asyncio.to_thread(sleeper))
    await asyncio.sleep(1)
    print("[BROKEN] Main function returning (but thread still sleeping)...")
    # When this returns, asyncio.run() will try to shut down and wait for thread
    # Result: HANGS for 100 seconds
    return


# ============================================================================
# FIX 1: Use daemon threads (threads die when main program exits)
# ============================================================================
async def main_fix1_daemon():
    print("\n[FIX1] Using daemon threads...")
    import threading

    def sleeper_with_daemon_check():
        print(f"Thread: Is daemon? {threading.current_thread().daemon}")
        time.sleep(100)

    # Create custom executor
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="daemon-worker")

    # Monkey-patch to make threads daemon
    # This is what we need to do in Flyte's LocalController
    original_submit = executor.submit

    def daemon_submit(fn, *args, **kwargs):
        fut = original_submit(fn, *args, **kwargs)
        # Set daemon on the thread
        for thread in executor._threads:
            thread.daemon = True
        return fut

    executor.submit = daemon_submit

    loop = asyncio.get_event_loop()
    f = loop.run_in_executor(executor, sleeper_with_daemon_check)

    await asyncio.sleep(1)
    print("[FIX1] Main returning (daemon thread will be abandoned)...")
    executor.shutdown(wait=False)
    return


# ============================================================================
# FIX 2: Add timeout to blocking calls (BEST for Flyte)
# ============================================================================
async def main_fix2_timeout():
    print("\n[FIX2] Using timeout on blocking call...")

    future = concurrent.futures.Future()

    def worker_with_timeout():
        """Simulates _task.py:319 - fut.result()"""
        try:
            # ORIGINAL (BROKEN): result = future.result()
            # FIX: Add timeout so it can be interrupted
            print("Thread: Waiting on future.result(timeout=2)...")
            result = future.result(timeout=2.0)
            return result
        except concurrent.futures.TimeoutError:
            print("Thread: Timeout! Returning None instead of hanging forever")
            return None

    f = asyncio.create_task(asyncio.to_thread(worker_with_timeout))
    await asyncio.sleep(1)
    print("[FIX2] Main returning (thread will timeout and exit gracefully)...")

    # Simulate informer failure - future never gets set
    # But timeout ensures thread exits within 2 seconds
    return


# ============================================================================
# FIX 3: Fire completion events on failure (BEST for Flyte Controller)
# ============================================================================
async def main_fix3_fire_events():
    print("\n[FIX3] Firing completion events when informer fails...")

    future = concurrent.futures.Future()

    def worker_waiting_for_completion():
        """Simulates submit_action waiting for informer to complete"""
        print("Thread: Waiting for future.result()...")
        result = future.result()
        print(f"Thread: Got result: {result}")
        return result

    f = asyncio.create_task(asyncio.to_thread(worker_waiting_for_completion))
    await asyncio.sleep(1)
    print("[FIX3] Informer failed! Firing completion event with error...")

    # FIX: Instead of leaving the future hanging, set an exception
    # This is what _bg_watch_for_errors should do
    future.set_exception(Exception("Informer failed!"))

    # Now the thread can complete
    try:
        await f
    except Exception as e:
        print(f"[FIX3] Task raised exception (as expected): {e}")

    print("[FIX3] Main returning (thread already completed)...")
    return


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Flyte Hang Demonstration and Fixes")
    print("=" * 70)

    if len(sys.argv) > 1 and sys.argv[1] == "--show-hang":
        print("\n⚠️  WARNING: This will hang for 100 seconds!")
        print("Press Ctrl+C to interrupt\n")
        asyncio.run(main_broken())
    else:
        print("\n(Run with --show-hang to see the broken version)")

    # Test all fixes
    print("\n" + "=" * 70)
    print("Testing Fix 1: Daemon Threads")
    print("=" * 70)
    asyncio.run(main_fix1_daemon())
    print("✅ Fix 1 completed without hanging!\n")

    print("=" * 70)
    print("Testing Fix 2: Timeout on Blocking Calls")
    print("=" * 70)
    asyncio.run(main_fix2_timeout())
    print("✅ Fix 2 completed without hanging!\n")

    print("=" * 70)
    print("Testing Fix 3: Fire Completion Events on Failure")
    print("=" * 70)
    asyncio.run(main_fix3_fire_events())
    print("✅ Fix 3 completed without hanging!\n")

    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print("Fix 1 (Daemon): Threads abandoned on exit (data loss risk)")
    print("Fix 2 (Timeout): Threads exit gracefully after timeout")
    print("Fix 3 (Events): Best for Flyte - properly signals failure")
    print("\nRecommended for Flyte: Combine Fix 2 + Fix 3")
