"""Runs an orchestrator on this machine with the remote worker's value semantics.

The sandbox bridge hands each value crossing between the source and the host to
a marshal. This one converts every value through a Flyte literal with
`_values`, the worker's rules, so the source sees what it would see on a
cluster: the same values, the same `TypeError`s for arguments that do not fit,
and a task's failure raised where it is called, as `RuntimeError("<code>:
<message>")`, where the source can catch it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping

from flyte.sandbox._bridge import RaiseInSandbox, _Marshal

from . import _values

if TYPE_CHECKING:
    from flyteidl2.core import interface_pb2, literals_pb2

    from flyte._task import TaskTemplate
    from flyte.models import NativeInterface


def _typed(interface: NativeInterface) -> interface_pb2.TypedInterface:
    from flyte._internal.runtime.types_serde import transform_native_to_typed_interface

    typed = transform_native_to_typed_interface(interface)
    assert typed is not None
    return typed


class LiteralBoundary(_Marshal):
    """The values of one orchestrator run, converted as the remote worker converts them."""

    # The worker answers only the orchestrator's tasks and conditions: any other call,
    # `flyte_map` included, is a NameError in the source.
    builtin_map = False

    def __init__(self, interface: NativeInterface, tasks: Mapping[str, Any]) -> None:
        super().__init__()
        self._interface = interface
        self._tasks = tasks

    def unknown_function(self, name: str) -> BaseException:
        return RaiseInSandbox(NameError(f"name '{name}' is not defined"))

    async def enter(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """The orchestrator's inputs, converted to literals and read as the worker reads them."""
        import flyte.errors
        from flyte.types import TypeEngine

        values: Dict[str, Any] = {}
        for name, (python_type, _) in self._interface.inputs.items():
            if name not in inputs:
                raise flyte.errors.RuntimeUserError("TypeError", f"missing the input '{name}'")
            literal_type = TypeEngine.to_literal_type(python_type)
            literal = await TypeEngine.to_literal(inputs[name], python_type, literal_type)
            try:
                values[name] = _values.to_monty(_values.from_literal(literal))
            except _values.ConversionError as error:
                raise flyte.errors.RuntimeUserError("TypeError", f"input '{name}': {error}") from None
        return values

    async def call(self, name: str, fn: Callable[..., Any], args: List[Any], kwargs: Dict[str, Any]) -> Any:
        from flyte._task import TaskTemplate

        task = self._tasks.get(name)
        if not isinstance(task, TaskTemplate):
            # A condition's host call, or a remote task, which cannot run on this machine.
            return await super().call(name, fn, args, kwargs)
        try:
            passed = [_values.from_monty(arg) for arg in args]
            passed_kwargs = [(key, _values.from_monty(value)) for key, value in kwargs.items()]
        except _values.UnsupportedType as error:
            raise RaiseInSandbox(TypeError(str(error))) from None
        try:
            return await self._call_task(task, passed, passed_kwargs)
        except _values.CallError as error:
            raise RaiseInSandbox(RuntimeError(f"{error.code}: {error}")) from None

    async def _call_task(self, task: TaskTemplate, args: List[Any], kwargs: List[Any]) -> Any:
        import flyte.errors
        from flyte._internal.runtime.convert import convert_from_native_to_error, convert_from_native_to_outputs
        from flyte.types import TypeEngine

        interface = task.native_interface
        typed = _typed(interface)
        literals = _values.bind_inputs(
            task.name,
            typed.inputs.variables,
            list(interface.inputs),
            await _default_literals(interface),
            args,
            kwargs,
        )
        native = {
            name: await TypeEngine.to_python_value(literal, interface.inputs[name][0]) for name, literal in literals
        }
        try:
            result = await task.aio(**native)
        except (RaiseInSandbox, flyte.errors.RemoteTaskUsageError):
            raise
        except Exception as exc:
            # Raised in the source as the worker raises a child's failure: the code and message
            # the task reports.
            error = convert_from_native_to_error(exc).err
            raise RaiseInSandbox(RuntimeError(f"{error.code}: {error.message}")) from exc
        if not interface.outputs:
            return None
        outputs = await convert_from_native_to_outputs(result, interface, task.name)
        named = [
            (output.name, output.value if output.HasField("value") else None)
            for output in outputs.proto_outputs.literals
        ]
        return _values.to_monty(_values.read_outputs(task.name, typed.outputs.variables, named))

    async def leave(self, output: Any) -> Any:
        """The source's result as the orchestrator's outputs, or a `TypeError` failing the run."""
        import flyte.errors
        from flyte.types import TypeEngine

        try:
            value = _values.from_monty(output)
        except _values.UnsupportedType as error:
            raise flyte.errors.RuntimeUserError("TypeError", str(error)) from None
        if not self._interface.outputs:
            return None
        typed = _typed(self._interface)
        try:
            literals = _values.encode_outputs(typed.outputs.variables, value)
        except _values.CallError as error:
            raise flyte.errors.RuntimeUserError(error.code, str(error)) from None
        results = [
            await TypeEngine.to_python_value(literal, self._interface.outputs[name]) for name, literal in literals
        ]
        return results[0] if len(results) == 1 else tuple(results)


async def _default_literals(interface: NativeInterface) -> Dict[str, literals_pb2.Literal]:
    """The task's defaults as literals, as its spec carries them to the worker."""
    import inspect

    from flyte.types import TypeEngine

    defaults: Dict[str, literals_pb2.Literal] = {}
    for name, (python_type, default) in interface.inputs.items():
        if default is inspect.Parameter.empty:
            continue
        literal_type = TypeEngine.to_literal_type(python_type)
        defaults[name] = await TypeEngine.to_literal(default, python_type, literal_type)
    return defaults


def marshal_for(interface: NativeInterface, tasks: Mapping[str, Any]) -> Callable[[], _Marshal]:
    """A factory of `LiteralBoundary`s for the bridge, one per execution."""
    return lambda: LiteralBoundary(interface, tasks)
