"""A basic streamlit app that uses a custom domain."""

import flyte
import flyte.app

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("streamlit==1.41.1")

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# In this case we do not need to supply any app code
# as we are using the built-in Streamlit `hello` app.
app_env = flyte.app.AppEnvironment(
    name="streamlit-hello-custom-domain",
    image=image,
    command="streamlit hello --server.port 8080",
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    domain=flyte.app.Domain(subdomain="custom-subdomain"),
)


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(app_env)
    print(d[0])
