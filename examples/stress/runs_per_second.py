import asyncio
import time

from aiolimiter import AsyncLimiter

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="runs_per_second",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_debian_base(flyte_version="2.0.0b34").with_pip_packages("plotly", "kaleido", "numpy"),
)

downstream_env = flyte.TaskEnvironment(
    name="downstream",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=(5, 20),
        idle_ttl=60,
        concurrency=50,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages(
        "unionai-reuse==0.1.9",
    ),
)


@downstream_env.task
async def sleeper(x: int) -> int:
    await asyncio.sleep(5.0)
    return x


@env.task(report=True)
async def runs_per_second(max_rps: int = 50, n: int = 500):
    """
    Measure actual runs per second while throttling above max_rps.
    Generates a comprehensive report with statistics and visualizations.
    """
    import plotly.graph_objects as go

    limiter = AsyncLimiter(max_rps, 1)

    start_time = time.time()
    run_times = []
    progress_data = []
    timestamps = []

    # Create report tabs
    overview_tab = flyte.report.get_tab("main")
    performance_tab = flyte.report.get_tab("Performance Charts")
    raw_data_tab = flyte.report.get_tab("Raw Data")

    overview_tab.log("<h2>🚀 Runs Per Second Calibration Test</h2>")
    overview_tab.log("<p><strong>Configuration:</strong></p>")
    overview_tab.log(f"<ul><li>Max RPS Setting: {max_rps}</li><li>Total Runs: {n}</li></ul>")
    overview_tab.log("<p><em>Starting test execution...</em></p>")
    await flyte.report.flush.aio()

    for i in range(n):
        run_start = time.time()

        async with limiter:
            # Create the run (this is the operation we're measuring)
            await flyte.run.aio(sleeper, x=i)

        run_end = time.time()
        run_time = run_end - run_start
        run_times.append(run_time)
        timestamps.append(run_end - start_time)

        # Collect progress data for plotting
        if (i + 1) % 10 == 0:  # Every 10 runs for smoother chart
            elapsed = run_end - start_time
            current_rps = (i + 1) / elapsed
            progress_data.append(
                {
                    "run_number": i + 1,
                    "elapsed_time": elapsed,
                    "current_rps": current_rps,
                    "avg_run_time_ms": sum(run_times[-10:]) / min(10, len(run_times)) * 1000,
                }
            )

        # Update progress in report every 50 runs
        if (i + 1) % 50 == 0:
            elapsed = run_end - start_time
            current_rps = (i + 1) / elapsed
            avg_run_time = sum(run_times[-50:]) / min(50, len(run_times))

            overview_tab.log(
                f"<p>✅ Completed {i + 1} runs: {current_rps:.2f} RPS | Avg run time: {avg_run_time * 1000:.2f}ms</p>"
            )
            await flyte.report.flush.aio()

    # Calculate final statistics
    total_time = time.time() - start_time
    actual_rps = n / total_time
    avg_run_time = sum(run_times) / len(run_times)
    min_run_time = min(run_times)
    max_run_time = max(run_times)
    theoretical_max_rps = 1 / min_run_time

    # Generate comprehensive report
    overview_tab.log("<hr><h3>📊 Final Results</h3>")
    results_table = f"""
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Runs</td><td>{n}</td></tr>
        <tr><td>Total Time</td><td>{total_time:.2f}s</td></tr>
        <tr><td>Max RPS Setting</td><td>{max_rps}</td></tr>
        <tr><td><strong>Actual RPS</strong></td><td><strong>{actual_rps:.2f}</strong></td></tr>
        <tr><td>Average Run Time</td><td>{avg_run_time * 1000:.2f}ms</td></tr>
        <tr><td>Min Run Time</td><td>{min_run_time * 1000:.2f}ms</td></tr>
        <tr><td>Max Run Time</td><td>{max_run_time * 1000:.2f}ms</td></tr>
        <tr><td>Theoretical Max RPS</td><td>{theoretical_max_rps:.2f}</td></tr>
    </table>
    """
    overview_tab.log(results_table)

    # Status analysis
    if actual_rps < max_rps * 0.9:
        status_msg = (
            f"⚠️ <strong>Throttling was NOT the limiting factor</strong><br>System can handle "
            f"~{actual_rps:.0f} RPS sequentially"
        )
        status_color = "orange"
    else:
        status_msg = f"✅ <strong>Successfully throttled at {max_rps} RPS</strong>"
        status_color = "green"

    overview_tab.log(
        f'<div style="background-color: {status_color}; padding: 10px; border-radius: 5px; color: white;">'
        f"{status_msg}</div>"
    )

    # Create performance charts
    if progress_data:
        # RPS over time chart
        rps_fig = go.Figure()
        rps_fig.add_trace(
            go.Scatter(
                x=[d["run_number"] for d in progress_data],
                y=[d["current_rps"] for d in progress_data],
                mode="lines+markers",
                name="Actual RPS",
                line={"color": "blue", "width": 2},
            )
        )
        rps_fig.add_hline(y=max_rps, line_dash="dash", line_color="red", annotation_text=f"Max RPS Limit ({max_rps})")
        rps_fig.update_layout(
            title="Runs Per Second Over Time", xaxis_title="Run Number", yaxis_title="RPS", height=400
        )

        # Run time distribution histogram
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=[rt * 1000 for rt in run_times], nbinsx=30, name="Run Time Distribution", marker_color="lightblue"
            )
        )
        hist_fig.update_layout(
            title="Run Time Distribution", xaxis_title="Run Time (ms)", yaxis_title="Frequency", height=400
        )

        # Run time over time chart
        timeline_fig = go.Figure()
        timeline_fig.add_trace(
            go.Scatter(
                x=list(range(1, len(run_times) + 1)),
                y=[rt * 1000 for rt in run_times],
                mode="lines",
                name="Run Time",
                line={"color": "green", "width": 1},
            )
        )
        timeline_fig.update_layout(
            title="Run Time Over Time", xaxis_title="Run Number", yaxis_title="Run Time (ms)", height=400
        )

        # Add charts to performance tab
        performance_tab.log("<h3>📈 Performance Visualizations</h3>")
        performance_tab.log(rps_fig.to_html(include_plotlyjs=True))
        performance_tab.log(hist_fig.to_html(include_plotlyjs=True))
        performance_tab.log(timeline_fig.to_html(include_plotlyjs=True))

    # Raw data tab
    raw_data_tab.log("<h3>📋 Raw Performance Data</h3>")
    raw_data_tab.log(f"<p>Run times (first 100): {[f'{rt * 1000:.2f}ms' for rt in run_times[:100]]}</p>")
    raw_data_tab.log("<p>Progress checkpoints:</p>")
    for data_point in progress_data[-10:]:  # Last 10 checkpoints
        raw_data_tab.log(
            f"<p>Run {data_point['run_number']}: {data_point['current_rps']:.2f} RPS, "
            f"{data_point['avg_run_time_ms']:.2f}ms avg</p>"
        )

    await flyte.report.flush.aio()


@env.task
async def main(n: int, max_per_n: int):
    await runs_per_second.override(short_name="primer")(max_rps=1, n=1)
    coros = []
    for run_number in range(n):
        coros.append(runs_per_second(max_rps=10, n=max_per_n))
    await asyncio.gather(*coros)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, n=10, max_per_n=10)
    print(run.url)
