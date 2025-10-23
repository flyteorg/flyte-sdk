import asyncio
from typing import Any, Dict

from flyte.syncify import syncify

from ._compiler import DAG, END_NODE, START_NODE

_SENTINEL_NODES = {START_NODE, END_NODE}


@syncify
async def run(dag: DAG, **kwargs) -> Any:
    """
    Execute a compiled DAG by running nodes in topological order.
    Independent nodes are executed in parallel using asyncio.gather.

    Args:
        dag: The compiled DAG to execute
        **kwargs: Workflow input parameters mapped to start node outputs

    Returns:
        The final output value(s) from the workflow
    """
    import networkx as nx

    print(f"Executing DAG with nodes: {list(dag.graph.nodes)}")
    # Storage for node outputs
    node_outputs: Dict[str, Dict[str, Any]] = {}

    # Initialize start node outputs with workflow inputs
    start_outputs = {}
    for i, (param_name, value) in enumerate(kwargs.items()):
        start_outputs[f"o{i}"] = value
    node_outputs[START_NODE] = start_outputs

    # Get topological order of nodes (excluding START_NODE and END_NODE)
    try:
        topo_order = list(nx.topological_sort(dag.graph))
    except nx.NetworkXError as e:
        raise ValueError(f"DAG contains a cycle: {e}")

    # Remove START_NODE and END_NODE from execution order
    execution_order = [node_id for node_id in topo_order if node_id not in _SENTINEL_NODES]

    # Track which nodes are ready to execute (all dependencies satisfied)
    # Mark START_NODE and END_NODE as already executed since they don't need actual execution
    executed = _SENTINEL_NODES

    async def execute_node(node_id: str):
        """Execute a single node and store its outputs."""
        node_data = dag.graph.nodes[node_id]
        node = node_data["node"]
        task = node.task

        if task is None:
            raise ValueError(f"Node {node_id} has no task to execute")

        # Collect inputs from predecessor nodes
        task_inputs = {}
        # For MultiDiGraph, we need to iterate over all edges (including multiple edges between same nodes)
        for predecessor, _, edge_data in dag.graph.in_edges(node_id, data=True):
            edge = edge_data["edge"]

            # Get the output from the predecessor node
            if predecessor not in node_outputs:
                raise ValueError(f"Predecessor {predecessor} has not been executed yet")

            predecessor_outputs = node_outputs[predecessor]
            if edge.output_name not in predecessor_outputs:
                raise ValueError(f"Output {edge.output_name} not found in node {predecessor}")

            # Map the output to the input parameter name
            task_inputs[edge.input_name] = predecessor_outputs[edge.output_name]

        print(f"Executing node {node_id}, {task.name} with inputs: {task_inputs}")
        # Execute the task using forward() for local execution
        result = task(**task_inputs)

        # Store outputs
        outputs = {}
        if task.interface.has_outputs():
            if len(task.interface.outputs) == 1:
                # Single output
                output_name = next(iter(task.interface.outputs.keys()))
                outputs[output_name] = result
            else:
                # Multiple outputs (tuple)
                for i, (output_name, _) in enumerate(task.interface.outputs.items()):
                    outputs[output_name] = result[i]

        node_outputs[node_id] = outputs

    print(f"Topological execution order: {execution_order}")
    # Execute nodes in topological order, parallelizing where possible
    while len(executed) < len(topo_order):
        # Find all nodes ready to execute (all predecessors executed)
        ready_nodes = []
        for node_id in execution_order:
            if node_id in executed:
                continue

            # Check if all predecessors have been executed
            predecessors = set(dag.graph.predecessors(node_id))
            if predecessors.issubset(executed):
                ready_nodes.append(node_id)

        print(f"Executing {len(ready_nodes)} nodes")
        if not ready_nodes:
            # No nodes ready - check if we're done or stuck
            if len(executed) == len(topo_order):
                break
            raise ValueError("DAG execution stuck - no nodes ready but not all executed")

        # Execute all ready nodes in parallel
        await asyncio.gather(*[execute_node(node_id) for node_id in ready_nodes])

        # Mark nodes as executed
        executed.update(ready_nodes)

    # Collect final outputs from END_NODE predecessors
    final_outputs = []
    for predecessor, _, edge_data in dag.graph.in_edges(END_NODE, data=True):
        edge = edge_data["edge"]

        predecessor_outputs = node_outputs[predecessor]
        if edge.output_name in predecessor_outputs:
            final_outputs.append(predecessor_outputs[edge.output_name])

    # Return the final result
    if len(final_outputs) == 0:
        return None
    elif len(final_outputs) == 1:
        return final_outputs[0]
    else:
        return tuple(final_outputs)
