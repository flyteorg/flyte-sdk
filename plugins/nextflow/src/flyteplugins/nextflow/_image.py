from __future__ import annotations

from pathlib import Path

import flyte

# Nextflow 25.10.6 bundles nf-amazon 3.4.5, the version nf-flyte is built against
DEFAULT_NEXTFLOW_VERSION = "25.10.6"
DEFAULT_NF_FLYTE_VERSION = "0.1.0"

NXF_HOME = "/opt/nxf"
_PLUGIN_ZIP_DIR = "/opt/nf-flyte"


def nextflow_image(
    *,
    nextflow_version: str = DEFAULT_NEXTFLOW_VERSION,
    nf_flyte: str | Path = DEFAULT_NF_FLYTE_VERSION,
    base: flyte.Image | None = None,
    name: str = "nextflow",
) -> flyte.Image:
    """
    Image for the task that runs the Nextflow head: a JRE, Nextflow and the nf-flyte plugin.

    Pipeline processes don't run in this image; each one runs in its own `container`.

    :param nextflow_version: Nextflow release to install.
    :param nf_flyte: nf-flyte plugin to install, either a version from the Nextflow plugin
        registry or the path to a locally built plugin zip (`nf-flyte-<version>.zip`).
    :param base: Image to add Nextflow to (default: the Flyte debian base image).
    :param name: Image name.
    """
    image = base or flyte.Image.from_debian_base(name=name)
    image = image.with_apt_packages("openjdk-17-jre-headless", "curl", "unzip").with_env_vars(
        {"NXF_HOME": NXF_HOME, "NXF_VER": nextflow_version, "NXF_ANSI_LOG": "false"}
    )

    if isinstance(nf_flyte, Path):
        version = _plugin_version_from_zip(nf_flyte)
        image = image.with_source_file(nf_flyte, f"{_PLUGIN_ZIP_DIR}/")
        install_plugin = [
            f"mkdir -p {NXF_HOME}/plugins/nf-flyte-{version}",
            f"unzip -q {_PLUGIN_ZIP_DIR}/{nf_flyte.name} -d {NXF_HOME}/plugins/nf-flyte-{version}",
        ]
    else:
        version = nf_flyte
        install_plugin = [f"nextflow plugin install nf-flyte@{version}"]

    return image.with_env_vars({"NF_FLYTE_VERSION": version}).with_commands(
        [
            "curl -fsSL https://get.nextflow.io | bash && mv nextflow /usr/local/bin/nextflow",
            # Download the Nextflow runtime and plugins at build time, not on every run. Without a
            # version, `plugin install` picks the release bundled with this Nextflow version (e.g.
            # nf-amazon 3.4.5 on 25.10.6, 3.9.2 on 26.04.x), not the newest in the registry.
            # nf-cloudcache keeps resume state in the work dir (see run_nextflow).
            "nextflow info && nextflow plugin install nf-amazon,nf-cloudcache",
            *install_plugin,
            f"chmod -R a+rwX {NXF_HOME}",
        ]
    )


def _plugin_version_from_zip(path: Path) -> str:
    prefix = "nf-flyte-"
    if not (path.name.startswith(prefix) and path.suffix == ".zip"):
        raise ValueError(f"Expected an nf-flyte plugin zip named 'nf-flyte-<version>.zip', got '{path.name}'")
    # no existence check: task modules are imported again inside the task pod, where the zip
    # doesn't exist. A missing zip fails the image build instead.
    return path.name[len(prefix) : -len(".zip")]
