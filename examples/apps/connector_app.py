"""A basic connector app that uses the Flyte BigQuery connector plugin."""

import flyte.app

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyteplugins-connectors[bigquery]", pre=True
)

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# In this case we do not need to supply any connector code
# as we are using the bigquery connector plugin.
app_env = flyte.app.ConnectorEnvironment(
    name="bigquery-connector-app",
    image=image,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(app_env)
    print(d[0])
