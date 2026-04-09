"""Test tailing logs from the Flyte Python SDK."""

import rich_click as click

import flyte
from flyte.remote import Run


@click.command()
@click.option("--endpoint", required=True, help="Flyte endpoint (e.g. dns:///...).")
@click.option("--project", required=True, help="Project name.")
@click.option("--domain", required=True, help="Domain name.")
@click.option("--run-id", required=True, help="Run ID to tail logs for.")
@click.option("--max-lines", default=50, show_default=True, type=int)
@click.option("--show-ts/--no-show-ts", default=True, show_default=True)
@click.option("--raw/--pretty", default=False, show_default=True)
def main(
    endpoint: str,
    project: str,
    domain: str,
    run_id: str,
    max_lines: int,
    show_ts: bool,
    raw: bool,
) -> None:
    flyte.init(endpoint=endpoint, project=project, domain=domain)

    run = Run.get(run_id)
    run.show_logs(max_lines=max_lines, show_ts=show_ts, raw=raw)


if __name__ == "__main__":
    main()
