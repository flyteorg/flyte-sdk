import re
from typing import Any, Dict, Optional

import yaml

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(
    "dynamic_dag",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)


# --- Dummy node functions (one per node type) ---


@env.task
async def node_1_fn(inputs: Dict[str, str]) -> str:
    print(f"node_1 running with inputs: {inputs}")
    return "result_from_node_1"


@env.task
async def node_2_fn(inputs: Dict[str, str]) -> str:
    print(f"node_2 running with inputs: {inputs}")
    return "result_from_node_2"


@env.task
async def node_3_fn(inputs: Dict[str, str]) -> str:
    print(f"node_3 running with inputs: {inputs}")
    return "result_from_node_3"


@env.task
async def node_4_fn(inputs: Dict[str, str]) -> str:
    print(f"node_4 running with inputs: {inputs}")
    return "result_from_node_4"


@env.task
async def node_5_fn(inputs: Dict[str, str]) -> str:
    print(f"node_5 running with inputs: {inputs}")
    return "result_from_node_5"


NODE_REGISTRY: Dict[str, Any] = {
    "node_1": node_1_fn,
    "node_2": node_2_fn,
    "node_3": node_3_fn,
    "node_4": node_4_fn,
    "node_5": node_5_fn,
}


def resolve_inputs(
    raw_inputs: Optional[Dict[str, str]],
    outputs: Dict[str, str],
) -> Dict[str, str]:
    """Resolve ${VAR_node_X} references in node inputs against collected outputs."""
    if not raw_inputs:
        return {}
    resolved = {}
    for key, value in raw_inputs.items():
        resolved[key] = re.sub(
            r"\$\{(\w+)\}",
            lambda m: outputs.get(m.group(1), m.group(0)),
            value,
        )
    return resolved


@env.task
async def run_pipeline(yaml_file: File, pipeline_name: str) -> Dict[str, str]:
    """Parse a YAML DAG config and execute the named pipeline.

    Nodes run sequentially. Skipped nodes are ignored.
    Outputs from earlier nodes can be forwarded to later nodes
    via the ${VAR_node_X} syntax in `inputs`.
    """
    async with yaml_file.open("rb") as f:
        raw = bytes(await f.read())
        # Replace non-breaking spaces (U+00A0) with regular spaces so YAML
        # indentation is parsed correctly regardless of how the file was authored.
        text = raw.decode("utf-8").replace("\u00a0", " ")
        config = yaml.safe_load(text)

    pipeline = config[pipeline_name]
    outputs: Dict[str, str] = {}

    # Nodes are ordered by their key (node_1, node_2, ...).
    sorted_nodes = sorted(pipeline.items(), key=lambda kv: kv[0])

    step = 0
    for node_name, node_cfg in sorted_nodes:
        if node_cfg.get("skip", False):
            print(f"Skipping {node_name}", flush=True)
            continue

        print(f"Running {node_name}", flush=True)
        node_fn = NODE_REGISTRY[node_name]
        raw_inputs = node_cfg.get("inputs")
        resolved = resolve_inputs(raw_inputs, outputs)

        with flyte.group(f"step-{step}-{node_name}"):
            result = await node_fn(inputs=resolved)

        output_var = node_cfg.get("output")
        if output_var:
            outputs[output_var] = result
        print(f"Finished {node_name} -> {output_var}={result}")
        step += 1

    return outputs


if __name__ == "__main__":
    flyte.init_from_config()
    yaml_file = File.from_local_sync("testyaml/dag2.yaml")
    run = flyte.with_runcontext("remote").run(
        run_pipeline,
        yaml_file=yaml_file,
        pipeline_name="pipeline_skip_1_2_3",
    )
    print(run.url)
    run.wait()
