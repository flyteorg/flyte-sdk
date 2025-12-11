import os
from pathlib import Path
from flyte._utils.v2_module_loader import ModuleLoader


def test_1_flat_directory():
    os.chdir("/Users/ytong/go/src/github.com/flyteorg/flyte-sdk/tests/project_structures/1_flat_directory")
    cwd = os.getcwd()
    print("")
    print(f"Current working directory: {cwd}")
    load_dir = Path(cwd)
    ml = ModuleLoader(load_dir, verbose=True)
    ml.compute_import_plan()
    modules_to_load = ml.discover_modules()
    # should be just foo and bar
    print(modules_to_load)
    assert len(modules_to_load) == 2
    assert set(modules_to_load) == {"foo", "bar"}
    assert ml.package_prefix == ""

