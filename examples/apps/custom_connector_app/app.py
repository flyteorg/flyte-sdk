"""Deploy the custom batch job connector as a long-running service."""

from pathlib import Path

import flyte
import flyte.app

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyte[connector]")

connector = flyte.app.ConnectorEnvironment(
    name="batch-job-connector",
    image=image,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    include=["my_connector"],
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    d = flyte.deploy(connector)
    print(d[0])
