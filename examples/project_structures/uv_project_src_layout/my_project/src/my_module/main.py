import logging
import pathlib

import flyte
from my_module.workflows.b import process_list

if __name__ == "__main__":
    # root_dir must point to src/ so modules are resolved as `my_module.*`
    # rather than `src.my_module.*`, preventing the same file from being
    # imported twice under different names.
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent.parent,
        log_level=logging.DEBUG,
    )

    run = flyte.run(process_list, x_list=list(range(10)))

    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")
