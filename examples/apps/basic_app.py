"""A basic app that uses the built-in Streamlit `hello` app.

Usage:
```
flyte -c ../../config.yaml deploy
```
"""

import flyte
import flyte.app

image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_pip_packages("union-runtime>=0.1.11", "streamlit==1.41.1")
    .with_pip_packages("flyte", pre=True, extra_args="--prerelease=allow")
)

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# In this case we do not need to supply any app code
# as we are using the built-in Streamlit `hello` app.
app = flyte.app.AppEnvironment(
    name="streamlit-hello-v2-004",
    image=image,
    args="streamlit hello --server.port 8080",
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)
