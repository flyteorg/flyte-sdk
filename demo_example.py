import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    def _adjust_thread_count(self):
        # Override to create daemon threads
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = '%s_%d' % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(name=thread_name, target=self._worker,
                                 args=(weakref_cb,), daemon=True)
            t.start()
            self._threads.add(t)


def sleeper():
    time.sleep(100)
    print("Sleeper finished")
    return "some valuable output"


async def main():
    executor = DaemonThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()
    f = loop.run_in_executor(executor, sleeper)

    await asyncio.sleep(1)
    print("Hello, World!")

    # Try to get result if completed, otherwise it gets killed on exit
    if f.done():
        result = await f
        print(f"Captured output: {result}")

    return

if __name__ == "__main__":
    asyncio.run(main())