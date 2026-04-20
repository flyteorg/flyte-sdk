"""A basic app that uses the built-in Streamlit `hello` app."""

import flyte
import flyte.app

image = flyte.Image.from_base("ghcr.io/flyteorg/flyte-demo:nightly").clone(name="flyte-oss-binary", extendable=True).with_pip_packages("flyte")

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# In this case we do not need to supply any app code
# as we are using the built-in Streamlit `hello` app.
app_env = flyte.app.AppEnvironment(
    name="flyte-oss-binary",
    image=image,
    # args="streamlit hello --server.port 8080",
    resources=flyte.Resources(cpu="8", memory="8Gi"),
)


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(app_env)
    print(d[0])
