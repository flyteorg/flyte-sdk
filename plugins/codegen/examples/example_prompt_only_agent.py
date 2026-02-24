"""Example: Log file analysis with sample data and schema context (Agent SDK approach).

Demonstrates:
- Passing a File as sample data (sampled by LLM for context)
- Using schema to describe input format and output structure
- Using constraints for validation rules
- Additional non-data inputs (filter_level: str)
- Multiple outputs (str, int, bool)
- Custom system_prompt override
- Using run() to run with the same sample data
"""

import logging
import tempfile
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.io import File
from flyte.sandbox import sandbox_environment

from flyteplugins.codegen import AutoCoderAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("flyteplugins.codegen.auto_coder_agent").setLevel(logging.INFO)

agent = AutoCoderAgent(
    name="log-parser-agent",
    model="claude-sonnet-4-5-20250929",
    use_agent_sdk=True,
)

env = flyte.TaskEnvironment(
    name="prompt-only-agent-example",
    secrets=[
        flyte.Secret(key="samhita_anthropic_api_key", as_env_var="ANTHROPIC_API_KEY"),
    ],
    resources=flyte.Resources(cpu=2, memory="5Gi"),
    image=(
        flyte.Image.from_debian_base()
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-codegen",
                pre=True,
            ),
        )
        .with_apt_packages("git")
        .with_pip_packages(
            "git+https://github.com/flyteorg/flyte-sdk.git@86f88fece16d956e28667d3f0d8d49108c8cdd68"
        )
    ),
    depends_on=[sandbox_environment],
)


@env.task(cache="auto")
def create_log_file() -> File:
    """Create a sample log file for processing."""
    log_content = """2024-01-15 08:23:01 INFO  [auth] User login successful user_id=1001
2024-01-15 08:23:15 ERROR [db] Connection timeout after 30s host=db-primary.internal
2024-01-15 08:23:16 WARN  [db] Failover to secondary host=db-secondary.internal
2024-01-15 08:23:17 INFO  [db] Connection established host=db-secondary.internal
2024-01-15 08:24:00 INFO  [api] GET /api/v1/users 200 45ms
2024-01-15 08:24:05 ERROR [api] POST /api/v1/orders 500 120ms error="inventory check failed"
2024-01-15 08:24:10 INFO  [api] GET /api/v1/products 200 32ms
2024-01-15 08:25:00 ERROR [auth] Failed login attempt user_id=unknown ip=192.168.1.50
2024-01-15 08:25:01 WARN  [auth] Rate limit approaching ip=192.168.1.50
2024-01-15 08:25:30 ERROR [api] GET /api/v1/users 503 5002ms error="upstream timeout"
2024-01-15 08:26:00 INFO  [auth] User login successful user_id=1002
2024-01-15 08:26:15 INFO  [api] GET /api/v1/dashboard 200 88ms
2024-01-15 08:27:00 ERROR [db] Deadlock detected transaction_id=TX-9921
2024-01-15 08:27:01 WARN  [db] Transaction rolled back transaction_id=TX-9921
2024-01-15 08:28:00 INFO  [api] POST /api/v1/orders 201 150ms
"""
    log_path = Path(tempfile.gettempdir()) / "app.log"
    log_path.write_text(log_content)
    return File.from_local_sync(str(log_path))


@env.task
async def parse_and_summarize_logs_with_agent(prompt: str, log_file: File) -> dict[str, str | int | bool]:
    """Generate log parsing code from prompt + schema using Agent SDK."""
    result = await agent.generate.aio(
        prompt=prompt,
        schema="""Output JSON schema for report_json:
{
    "time_range": {"start": "ISO timestamp", "end": "ISO timestamp"},
    "total_lines": int,
    "by_level": {"INFO": int, "WARN": int, "ERROR": int},
    "by_component": {"component_name": {"info": int, "warn": int, "error": int}},
    "error_messages": ["list of error message strings"]
}""",
        constraints=[
            "Must handle all log levels: INFO, WARN, ERROR",
            "Must not crash on malformed lines - skip them with a warning",
            "Component names must be extracted from square brackets",
            "worst_component is the component with the highest error count",
            "filter_level controls minimum severity: INFO shows all, WARN shows WARN+ERROR, ERROR shows only ERROR",
        ],
        samples={"log_file": log_file},
        inputs={"filter_level": str},
        outputs={
            "report_json": str,
            "total_errors": int,
            "worst_component": str,
            "has_critical_errors": bool,
        },
    )

    if not result.success:
        return {"error": result.error, "attempts": result.attempts}

    report_json, total_errors, worst_component, has_critical_errors = await result.run.aio(
        name="run_log_parser_agent", filter_level="WARN"
    )

    return {
        "report_json": report_json,
        "total_errors": total_errors,
        "worst_component": worst_component,
        "has_critical_errors": has_critical_errors,
    }


@env.task
async def log_analysis_agent_workflow(
    prompt: str = """Parse application log files and produce a structured health report.

Each log line follows the format: TIMESTAMP LEVEL [component] message key=value...

Only include lines at or above the filter_level severity (INFO < WARN < ERROR).
Parse each matching line, compute summary statistics, and identify the component
with the most errors (worst_component). Set has_critical_errors to true if any
ERROR lines exist. Output report_json as a JSON string with the summary.""",
) -> dict[str, str | int | bool]:
    """Parse logs with Agent SDK: prompt-only, explicit inputs/outputs."""
    log_file = create_log_file()
    return await parse_and_summarize_logs_with_agent(prompt=prompt, log_file=log_file)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(log_analysis_agent_workflow)
    print(f"\nRun URL: {run.url}")
