import asyncio
import time

from aiolimiter import AsyncLimiter

import flyte

env = flyte.TaskEnvironment(
    name="runs_per_second",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

downstream_env = flyte.TaskEnvironment(
    name="downstream",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=10,
        idle_ttl=60,
        concurrency=10,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base(install_flyte=False).with_pip_packages(
        "unionai-reuse==0.1.6b0", "flyte>2.0.0b21", pre=True
    ),
)


@downstream_env.task
async def sleeper(x: int) -> int:
    await asyncio.sleep(1.0)
    return x


@env.task
async def runs_per_second(max_rps: int = 50, n: int = 500):
    """
    Measure actual runs per second while throttling above max_rps.
    Prints statistics to calibrate sequential run creation performance.
    """
    limiter = AsyncLimiter(max_rps, 1)

    start_time = time.time()
    run_times = []

    print(f"Starting {n} runs with max RPS throttling at {max_rps}")
    print("=" * 60)

    for i in range(n):
        run_start = time.time()

        async with limiter:
            # Create the run (this is the operation we're measuring)
            await flyte.run.aio(sleeper, x=i)

        run_end = time.time()
        run_times.append(run_end - run_start)

        # Print progress every 50 runs
        if (i + 1) % 50 == 0:
            elapsed = run_end - start_time
            current_rps = (i + 1) / elapsed
            avg_run_time = sum(run_times[-50:]) / min(50, len(run_times))
            print(f"Runs {i + 1:3d}: {current_rps:6.2f} RPS | Avg run time: {avg_run_time * 1000:6.2f}ms")

    # Final statistics
    total_time = time.time() - start_time
    actual_rps = n / total_time
    avg_run_time = sum(run_times) / len(run_times)
    min_run_time = min(run_times)
    max_run_time = max(run_times)

    print("=" * 60)
    print("RESULTS:")
    print(f"  Total runs:        {n}")
    print(f"  Total time:        {total_time:.2f}s")
    print(f"  Max RPS setting:   {max_rps}")
    print(f"  Actual RPS:        {actual_rps:.2f}")
    print(f"  Avg run time:      {avg_run_time * 1000:.2f}ms")
    print(f"  Min run time:      {min_run_time * 1000:.2f}ms")
    print(f"  Max run time:      {max_run_time * 1000:.2f}ms")

    # Calculate theoretical max RPS without throttling
    theoretical_max_rps = 1 / min_run_time
    print(f"  Theoretical max:   {theoretical_max_rps:.2f} RPS (based on min run time)")

    if actual_rps < max_rps * 0.9:  # If we're significantly below the limit
        print("  Status: Throttling was NOT the limiting factor")
        print(f"          System can handle ~{actual_rps:.0f} RPS sequentially")
    else:
        print(f"  Status: Successfully throttled at {max_rps} RPS")

    print("=" * 60)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(runs_per_second, max_rps=50, n=500)
    print(run.url)
