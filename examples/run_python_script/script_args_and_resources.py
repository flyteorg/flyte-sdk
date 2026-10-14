"""
Run an arbitrary Python script remotely with custom resources, script
arguments, and an output directory.

This file is a **plain script**, not a Flyte task module: `flyte run
python-script` bundles it into a container and runs it as a subprocess, so
it just parses `sys.argv` like any other CLI script — the `--extra-args`
values below are handed straight to it as `argv`.

Run with 2 CPUs / 4Gi memory, two script args, and an output directory that
gets uploaded as a `flyte.io.Dir` once the script exits:

    flyte run python-script examples/run_python_script/script_args_and_resources.py \\
        --cpu 2 --memory 4Gi \\
        --extra-args "--epochs,5,--lr,0.01" \\
        --output-dir /tmp/outputs

Follow to completion and see the printed output + exit code:

    flyte run --follow python-script examples/run_python_script/script_args_and_resources.py \\
        --extra-args "--epochs,5,--lr,0.01" --output-dir /tmp/outputs
"""

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Running with epochs={args.epochs}, lr={args.lr}")
    print(f"CPU count visible to this container: {os.cpu_count()}")

    for epoch in range(args.epochs):
        loss = args.lr / (epoch + 1)
        print(f"epoch={epoch} loss={loss:.4f}")

    # `--output-dir` tells the framework which path (inside the container) to
    # upload after the script finishes — write to the *same* path you pass on
    # the CLI (there's no env var for this; it's just a literal shared path).
    output_dir = "/tmp/outputs"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.txt"), "w") as f:
        f.write(f"trained for {args.epochs} epochs at lr={args.lr}\n")
    print(f"Wrote results to {output_dir}")
