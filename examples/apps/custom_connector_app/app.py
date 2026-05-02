"""Deploy the custom batch job connector as a long-running service."""

from pathlib import Path

import flyte
import flyte.app

image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_apt_packages("git")
    .with_pip_packages("flyte[connector]")
    .with_pip_packages(
        "git+https://github.com/pingsutw/flyte.git@5c4b05b50fef993e4bbf36243a3bcd492fb604f5#subdirectory=gen/python"
    )
)

connector = flyte.app.ConnectorEnvironment(
    name="batch-job-connector123",
    image=image,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    include=["my_connector"],
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    d = flyte.deploy(connector)
    print(d[0])
