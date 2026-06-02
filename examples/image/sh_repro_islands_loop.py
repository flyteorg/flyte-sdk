"""Loop UNIQUE minimal builds against islands to catch the intermittent
base-config-resolve failure (sh: executable file not found in $PATH).

Each iteration uses a distinct command -> distinct image hash -> no build cache
hit -> the base image config is resolved fresh every time, which is the step
that intermittently flakes.
"""

import traceback

import flyte
from flyte import Image


def make_image(i: int) -> Image:
    return Image.from_debian_base(python_version=(3, 12), install_flyte=False).with_commands([f"echo hello {i}"])


if __name__ == "__main__":
    flyte.init_from_config(
        "/Users/ytong/go/src/github.com/unionai/cloud/gen/cli-config/uctl/islands.production_v2.yaml"
    )
    N = 15
    fails = 0
    for i in range(1, N + 1):
        img = make_image(i)
        try:
            result = flyte.build(img, force=True, wait=True)
            run = result.remote_run
            phase = getattr(run, "phase", "?")
            name = getattr(run, "name", "?")
            ok = "OK " if "SUCCEEDED" in str(phase) else "FAIL"
            if ok == "FAIL":
                fails += 1
            print(f"[{i:02d}/{N}] {ok} phase={phase} run={name} uri={result.uri.split(':')[-1][:16]}", flush=True)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"[{i:02d}/{N}] EXC {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"\n=== DONE: {fails}/{N} failed ===", flush=True)
