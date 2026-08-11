"""
Run a plain script on a Ray cluster via `flyte run python-script --plugin-config`.

`--plugin-config` points at a YAML file naming a fully-qualified plugin
config class — here `flyteplugins.ray.RayJobConfig`, see
`examples/run_python_script/ray_plugin_config.yaml` — plus its constructor
arguments. Flyte starts the Ray head + worker pods for the task and calls
`ray.init()` itself before this script runs; this plain script just attaches
to that already-running local Ray session with `ray.init(address="auto")`.

Run:

    flyte run python-script examples/run_python_script/ray_plugin_config.py \\
        --packages "ray[default]==2.46.0,flyteplugins-ray" \\
        --plugin-config examples/run_python_script/ray_plugin_config.yaml

Follow to completion:

    flyte run --follow python-script examples/run_python_script/ray_plugin_config.py \\
        --packages "ray[default]==2.46.0,flyteplugins-ray" \\
        --plugin-config examples/run_python_script/ray_plugin_config.yaml

Swap in a different plugin (PyTorch elastic, Spark, Databricks, or any other
fully-qualified plugin config dataclass) by pointing `--plugin-config` at a
YAML file with a different `plugin:`/`config:` — the CLI flag and the
recursive YAML-to-dataclass loading work the same way for all of them.
"""

import ray


@ray.remote
def square(x: int) -> int:
    return x * x


if __name__ == "__main__":
    ray.init(address="auto")
    futures = [square.remote(i) for i in range(10)]
    results = ray.get(futures)
    print(f"squares: {results}")
    assert results == [i * i for i in range(10)]
    print("ray plugin-config smoke test passed")
