"""
HTML Template Report Example.

Demonstrates `TaskEnvironment.include`: the Python code stays in this file,
while the HTML skeleton lives in `report_template.html` alongside it. The
template is shipped to the task's container via `include`, read at runtime,
and substituted with dynamic values before being logged to the report.

Run remotely:

    flyte run --follow examples/reports/html_template_report.py generate_template_report
"""

from datetime import datetime
from pathlib import Path

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="html_template_report",
    image=flyte.Image.from_debian_base(python_version=(3, 12)),
    include=("report_template.html",),
)


@env.task(report=True)
async def generate_template_report() -> str:
    template_path = Path(__file__).parent / "report_template.html"
    template = template_path.read_text()

    items = "\n    ".join(
        f"<li>{item}</li>"
        for item in (
            "Bundled via Environment.include",
            "Readable on the remote pod",
            "Rendered into the Flyte report UI",
        )
    )

    body = template.format(
        title="Hello from a bundled HTML file",
        greeting="This page was rendered from a separate HTML template that was shipped with the task.",
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        items=items,
    )

    flyte.report.get_tab("Main").log(body)
    await flyte.report.flush.aio()
    return "ok"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(generate_template_report)
    print(run.name)
    print(run.url)
