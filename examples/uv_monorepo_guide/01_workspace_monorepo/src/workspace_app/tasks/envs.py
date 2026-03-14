import pathlib

import flyte

WORKSPACE_ROOT = pathlib.Path(__file__).parent.parent.parent.parent  # -> 01_workspace_monorepo/

etl_env = flyte.TaskEnvironment(
    name="etl",
    resources=flyte.Resources(memory="512Mi", cpu="1"),
    image=flyte.Image.from_debian_base()
    .with_uv_project(
        pyproject_file=WORKSPACE_ROOT / "pyproject.toml",
        extra_args="--only-group etl",
    )
    .with_code_bundle(),
)

ml_env = flyte.TaskEnvironment(
    name="ml",
    resources=flyte.Resources(memory="1Gi", cpu="1"),
    image=flyte.Image.from_debian_base()
    .with_uv_project(
        pyproject_file=WORKSPACE_ROOT / "pyproject.toml",
        extra_args="--only-group ml",
    )
    .with_code_bundle(),
)
