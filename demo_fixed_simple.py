"""
Simple demonstration of the hang and two practical fixes for Flyte.
"""
import asyncio
import concurrent.futures
import time


def sleeper():
    """Simulates a blocking call that hangs for 100 seconds"""
    print("  → Thread started, sleeping for 100 seconds...")
    time.sleep(100)
    print("  → Thread woke up!")


# ============================================================================
# FIX 1: Add timeout to blocking future.result() calls
# ============================================================================
async def fix1_timeout():
    """
    This is the practical fix for _task.py:319
    Change: fut.result() → fut.result(timeout=X)
    """
    print("\n" + "=" * 70)
    print("FIX 1: Add timeout to fut.result()")
    print("=" * 70)

    future = concurrent.futures.Future()

    def worker():
        """Simulates Thread 33 blocked on fut.result()"""
        try:
            print("  → Thread calling future.result(timeout=2.0)...")
            result = future.result(timeout=2.0)  # ← THE FIX
            return result
        except concurrent.futures.TimeoutError:
            print("  → Thread timed out! Exiting gracefully.")
            return None

    task = asyncio.create_task(asyncio.to_thread(worker))
    await asyncio.sleep(0.5)

    print("  Main: Simulating informer failure (future never set)...")
    await asyncio.sleep(0.5)

    print("  Main: Returning (thread will timeout in ~1 second)...")
    # The thread will timeout and exit, allowing clean shutdown
    return


# ============================================================================
# FIX 2: Fire completion events when informer fails
# ============================================================================
async def fix2_fire_events():
    """
    This is the fix for _bg_watch_for_errors in _core.py
    When informer fails, fire all pending completion events
    """
    print("\n" + "=" * 70)
    print("FIX 2: Fire completion events on failure")
    print("=" * 70)

    future = concurrent.futures.Future()

    def worker():
        """Simulates Thread 33 waiting for completion"""
        try:
            print("  → Thread calling future.result()...")
            result = future.result()  # No timeout needed!
            return result
        except Exception as e:
            print(f"  → Thread got exception: {type(e).__name__}: {e}")
            return None

    task = asyncio.create_task(asyncio.to_thread(worker))
    await asyncio.sleep(0.5)

    print("  Main: Informer failed! Setting exception on future...")
    future.set_exception(RuntimeError("Informer watch failure"))  # ← THE FIX

    await asyncio.sleep(0.5)
    print("  Main: Returning (thread already completed)...")
    return


# ============================================================================
# BROKEN: Show the hang (only if requested)
# ============================================================================
async def show_hang():
    """Don't run this unless you want to wait 100 seconds!"""
    print("\n" + "=" * 70)
    print("BROKEN VERSION (will hang)")
    print("=" * 70)

    print("  Creating thread that will block for 100 seconds...")
    f = asyncio.create_task(asyncio.to_thread(sleeper))
    await asyncio.sleep(1)

    print("  Main returning...")
    print("  ⚠️  asyncio.run() cleanup will now hang waiting for thread...")
    # This will hang for 100 seconds!
    return


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Flyte Thread Hang - Demonstration and Fixes")
    print("=" * 70)

    if "--show-hang" in sys.argv:
        print("\n⚠️  WARNING: About to demonstrate the hang (100 seconds)")
        print("Press Ctrl+C to interrupt")
        input("Press Enter to continue...")
        asyncio.run(show_hang())
        print("If you see this, you waited 100 seconds!")
    else:
        print("\n✓ Skipping broken version (use --show-hang to see it)")

    # Test Fix 1
    start = time.time()
    asyncio.run(fix1_timeout())
    elapsed = time.time() - start
    print(f"✅ Fix 1 completed in {elapsed:.1f}s (should be ~3s)\n")

    # Test Fix 2
    start = time.time()
    asyncio.run(fix2_fire_events())
    elapsed = time.time() - start
    print(f"✅ Fix 2 completed in {elapsed:.1f}s (should be ~1s)\n")

    print("=" * 70)
    print("SUMMARY - How to fix Flyte")
    print("=" * 70)
    print()
    print("Fix 1: In _task.py line 319, change:")
    print("  x = fut.result(None)")
    print("  →  x = fut.result(timeout=60)")  # or some reasonable timeout
    print()
    print("Fix 2: In _core.py _bg_watch_for_errors(), add:")
    print("  await self._informers.fire_all_completion_events()")
    print("  before setting self._running = False")
    print()
    print("Recommended: Implement BOTH fixes for defense in depth")
