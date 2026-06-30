import pathlib

import flyte

MY_APP_ROOT = pathlib.Path(__file__).parent.parent.parent  # -> my_app/
MY_LIB_PKG = MY_APP_ROOT.parent / "my_lib" / "src" / "my_lib"  # -> my_lib/src/my_lib/

env = flyte.TaskEnvironment(
    name="my_app",
    resources=flyte.Resources(memory="256Mi", cpu="1"),
    # my_lib is an editable path dep in pyproject.toml (so uv_build can find its source
    # during image build). Its package files are also baked into the image at /root/my_lib/
    # via with_source_folder, so they're importable at runtime without relying on the
    # editable install's .pth file (which points to a build-stage-only path).
    image=flyte.Image.from_debian_base()
    .with_uv_project(pyproject_file=MY_APP_ROOT / "pyproject.toml")
    .with_source_folder(MY_LIB_PKG)
    .with_code_bundle(),
)
