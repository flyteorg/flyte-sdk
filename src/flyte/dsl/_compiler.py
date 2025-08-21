from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator

import networkx as nx
import rich.repr

from flyte._task import TaskTemplate

END_NODE = "end-node"

START_NODE = "start-node"

if TYPE_CHECKING:
    import pydot


@rich.repr.auto
@dataclass(kw_only=True)
class DAGNode:
    """
    A DAGNode represents a node in the DAG, which is a task that can be executed.
    It contains the task template and its unique ID.
    """

    id: str
    task: TaskTemplate | None

    def to_pydot(self) -> pydot.Node:
        """Convert this DAGNode to a pydot Node for visualization."""
        import pydot

        if self.task is None:
            raise RuntimeError("DAGNode must have a task to convert to pydot Node.")
        label = self.task.name
        return pydot.Node(self.id, label=label, shape="box")


@dataclass
class StartNode(DAGNode):
    """
    StartNode is a special node that represents the starting point of the DAG.
    It is a dummy node used to initialize the DAG.
    """

    id: str = START_NODE
    task: TaskTemplate | None = None  # Start node does not have a task

    def to_pydot(self) -> pydot.Node:
        """Convert this StartNode to a pydot Node for visualization."""
        import pydot

        label = "Start"
        return pydot.Node(self.id, label=label, shape="ellipse", style="filled", fillcolor="lightgrey")


@rich.repr.auto
@dataclass
class EndNode(DAGNode):
    """
    EndNode is a special node that represents the end point of the DAG.
    It is a dummy node used to finalize the DAG.
    """

    id: str = END_NODE
    task: TaskTemplate | None = None  # End node does not have a task

    def to_pydot(self) -> pydot.Node:
        """Convert this EndNode to a pydot Node for visualization."""
        import pydot

        label = "End"
        return pydot.Node(self.id, label=label, shape="ellipse", style="filled", fillcolor="lightgrey")


@rich.repr.auto
@dataclass
class DAGEdge:
    """
    A DAGEdge represents a directed edge in the DAG, which connects two DAGNodes.
    It contains the output name from the source node and the input name for the target node.
    """

    from_node_id: str
    to_node_id: str
    output_name: str | None = None
    input_name: str | None = None

    def to_pydot(self) -> pydot.Edge:
        """Convert this DAGEdge to a pydot Edge for visualization."""
        import pydot

        if self.output_name is None or self.input_name is None:
            return pydot.Edge(str(self.from_node_id), str(self.to_node_id))

        label = f"{self.output_name} → {self.input_name}"
        return pydot.Edge(str(self.from_node_id), str(self.to_node_id), label=label)


class DAG:
    """
    A DAG (Directed Acyclic Graph) represents a computational graph with tasks as nodes
    and data dependencies as edges.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.task_nodes: Dict[str, DAGNode] = {}  # Maps task_id to TaskTemplate instances
        self.graph.add_node(START_NODE, node=StartNode())

    def add_task(self, unique_id: str, task_template: TaskTemplate) -> str:
        """Add a task to the DAG and return its unique ID."""
        node = DAGNode(task=task_template, id=unique_id)
        self.task_nodes[unique_id] = node
        self.graph.add_node(unique_id, node=node)
        return unique_id

    def add_dependency(self, from_node_id: str, to_node_id: str, output_name: str, input_name: str):
        """Add a dependency edge between two tasks."""
        edge = DAGEdge(from_node_id=from_node_id, to_node_id=to_node_id, output_name=output_name, input_name=input_name)
        self.graph.add_edge(from_node_id, to_node_id, edge=edge)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the DAG."""
        return self.graph.number_of_nodes()

    @property
    def nodes(self) -> Iterator[DAGNode]:
        """Return the nodes in the DAG."""
        for node_id, data in self.graph.nodes(data=True):
            print(f"{node_id} = {data}")
            yield data["node"]

    @property
    def edges(self) -> Iterator[DAGEdge]:
        """Return the edges in the DAG."""
        for from_id, to_id, data in self.graph.edges(data=True):
            yield data["edge"]

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the DAG."""
        return self.graph.number_of_edges()

    def end(self):
        """Finalize the DAG by adding an end node."""
        end_node = EndNode()
        self.graph.add_node(END_NODE, node=end_node)

    def to_dot(self):
        """Convert the DAG to a pydot graph for visualization."""
        import pydot

        dot = pydot.Dot(graph_type="digraph", rankdir="TB")

        # Add nodes
        for task_id, node in self.task_nodes.items():
            # Use just the task name as the label for cleaner visualization
            dot.add_node(node.to_pydot())

        # Add edges
        for from_id, to_id, edge_data in self.graph.edges(data=True):
            dot.add_edge(edge_data["edge"].to_pydot())

        return dot


@rich.repr.auto
@dataclass
class Promise:
    """
    A Promise represents a future value that will be computed by a task.
    It tracks the task that produces the value and the output name.
    """

    from_node_id: str
    output_name: str


def compile(fn) -> DAG:
    """
    The `compile` function takes a Python function and replaces its parameters with `Promise` objects to build a DAG.
    The DAG uses `networkx` to store nodes (tasks) and edges (data dependencies).
    Each task, decorated with `@task`, defines its inputs and outputs, enabling automatic parallelization in the DAG.
    """

    dag = DAG()
    task_call_counter = 0  # Counter to ensure unique task IDs

    # Get the signature of the function to compile
    sig = inspect.signature(fn)

    # Create mock promises for the function's input parameters
    mock_inputs = {}
    i = 0
    for param_name, param in sig.parameters.items():
        promise = Promise(START_NODE, f"o{i}")
        i += 1
        mock_inputs[param_name] = promise

    # Monkey-patch TaskTemplate.__call__ to track calls during compilation
    original_call = TaskTemplate.__call__

    def tracking_call(self: TaskTemplate, *args, **kwargs):
        nonlocal task_call_counter
        task_call_counter += 1
        node_id = f"n-{task_call_counter}"

        # Convert args to kwargs using the interface
        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)

        # Add this task to the DAG
        node_id = dag.add_task(node_id, self)

        # Create dependencies for each input that is a Promise
        for input_name, value in kwargs.items():
            if isinstance(value, Promise):
                dag.add_dependency(value.from_node_id, node_id, value.output_name, input_name)
            elif isinstance(value, (int, float, str)):
                # Handle constant values - we can ignore them for now
                pass

        # Create promises for each output of this task
        if self.interface.has_outputs():
            if len(self.interface.outputs) == 1:
                # Single output
                output_name = next(iter(self.interface.outputs.keys()))
                promise = Promise(node_id, output_name)
                return promise
            else:
                # Multiple outputs - return a tuple of promises
                promises = []
                for output_name in self.interface.outputs.keys():
                    promise = Promise(node_id, output_name)
                    promises.append(promise)
                return tuple(promises)
        else:
            # No outputs
            return None

    # Apply the monkey patch
    setattr(TaskTemplate, "__call__", tracking_call)

    try:
        # Execute the function with mock inputs to trace the computation
        result = fn(**mock_inputs)

        dag.end()
        if result is not None:
            # If the function returns a value, we need to handle it
            if isinstance(result, Promise):
                # Single output promise
                dag.add_dependency(result.from_node_id, END_NODE, result.output_name, "o0")
            elif isinstance(result, tuple):
                # Multiple outputs - add dependencies for each
                for i, promise in enumerate(result):
                    if isinstance(promise, Promise):
                        dag.add_dependency(promise.from_node_id, END_NODE, promise.output_name, f"o{i}")

        return dag

    except TypeError as e:
        # Handle cases where non-@task functions are called with Promise objects
        # Only fail if the error is about iterating over promises or using promises as integers
        if (
            "cannot be interpreted as an integer" in str(e)
            or "'Promise' object is not iterable" in str(e)
            or "unsupported operand type(s)" in str(e)
        ):
            raise AssertionError(
                f"Function {fn.__name__} calls a non-@task function. All operations must use @task decorated functions."
            )
        else:
            # For other TypeErrors (like unsupported operand types), just return the DAG as-is
            # This allows functions that call non-@task functions to compile but may fail at runtime
            return dag
    except Exception as e:
        # Re-raise compilation errors
        raise e
    finally:
        # Restore the original __call__ method
        setattr(TaskTemplate, "__call__", original_call)
