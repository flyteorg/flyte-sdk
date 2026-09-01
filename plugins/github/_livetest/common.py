"""Shared image + secret wiring for live-testing the github plugin.

The plugin is not published to PyPI yet, so images are built from locally
built wheels in dist/ instead of `with_pip_packages("flyteplugins-github")`.
The cluster has GITHUB_PAT rather than GITHUB_TOKEN, so it is mounted under
the env var name the plugin expects.
"""

import flyte

GH_SECRET = flyte.Secret("GITHUB_PAT", as_env_var="GITHUB_TOKEN")
WEBHOOK_SECRET = flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET")

REPO = "cosmicBboy/toy-repo"


def image(*extra_packages: str) -> flyte.Image:
    img = (
        flyte.Image.from_debian_base(python_version=(3, 12))
        .with_local_v2()
        .with_local_v2_plugins("flyteplugins-github")
    )
    if extra_packages:
        img = img.with_pip_packages(*extra_packages)
    return img
