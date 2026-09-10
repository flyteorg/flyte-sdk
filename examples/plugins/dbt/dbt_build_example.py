from __future__ import annotations

import pathlib
from pathlib import Path

from flyteplugins.dbt import DbtNodeResult, DbtTask

import flyte

DBT_PROJECT_DIR = "jaffle_shop"
DBT_PROFILES_DIR = "dbt-profiles"

image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_requirements("requirements.txt")
    .with_source_folder(Path(DBT_PROJECT_DIR))
    .with_source_folder(Path(DBT_PROFILES_DIR))
)


def print_dbt_event(event):
    event_name = getattr(getattr(event, "info", None), "name", type(event).__name__)
    data = getattr(event, "data", None)
    node_info = getattr(data, "node_info", None)
    if node_info is None:
        return

    node_name = getattr(node_info, "node_name", None)
    node_status = getattr(node_info, "node_status", None)
    print(f"dbt event={event_name} node={node_name} status={node_status}")


env = flyte.TaskEnvironment(
    name="dbt-jaffle-shop",
    image=image,
)

dbt_build = DbtTask(
    name="dbt-build",
    task_environment=env,
    callbacks=[print_dbt_event],
)


def dbt_project_args(command: str) -> list[str]:
    return [
        command,
        "--project-dir",
        DBT_PROJECT_DIR,
        "--profiles-dir",
        DBT_PROFILES_DIR,
        "--profile",
        DBT_PROJECT_DIR,
    ]


@env.task
async def main() -> list[DbtNodeResult]:
    build_result = await dbt_build.aio(cli_args=dbt_project_args("build"))
    return build_result


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(main)
    print("run name:", run.name)
    print("run url:", run.url)
