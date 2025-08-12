import inspect
import json
import typing
from dataclasses import asdict, dataclass
from functools import partial

import agents
from agents import FunctionTool as OpenAIFunctionTool
from agents import function_tool as openai_function_tool
from agents.function_schema import function_schema
from agents.tool_context import ToolContext
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate
from flyte.models import NativeInterface
from packaging import version

MIN_PACKAGE_VERSION = "0.2.4"
assert version.parse(agents.__version__) >= version.parse(MIN_PACKAGE_VERSION), (
    f"The agents package needs to be at least version {MIN_PACKAGE_VERSION}, found version {agents.__version__}"
)


@dataclass
class FunctionTool(OpenAIFunctionTool):
    """
    Flyte-compatible replacement for agents.FunctionTool

    This is a dataclass that includes additional fields that are not present in
    the OpenAI FunctionTool dataclass so that the tool can be used as a flyte
    task.
    """

    task: TaskTemplate | None = None
    native_interface: NativeInterface | None = None
    report: bool = False

    async def execute(self, *args, **kwargs):
        return await self.task.execute(*args, **kwargs)


def function_tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    **kwargs,
) -> FunctionTool | OpenAIFunctionTool:
    """Flyte-compatible replacement for @agents.function_tool

    **kwargs are forwarded to the underlying @agents.function_tool decorator.
    For @flyte.trace functions, this just forwards all the arguments to the
    agents.function_tool decorator:
    https://openai.github.io/openai-agents-python/ref/tool/#agents.tool.function_tool

    For @TaskEnvironment.task functions, this will create a flyte-compatible
    FunctionTool dataclass that can run tools as flyte tasks.
    """
    if func is None:
        return partial(function_tool, **kwargs)

    if isinstance(func, AsyncFunctionTaskTemplate):

        async def _on_invoke_tool(ctx: ToolContext[typing.Any], input: str) -> typing.Any:
            json_data: dict[str, typing.Any] = json.loads(input) if input else {}
            schema = function_schema(func.func)
            parsed = schema.params_pydantic_model(**json_data) if json_data else schema.params_pydantic_model()
            args, kwargs_dict = schema.to_call_args(parsed)
            if inspect.iscoroutinefunction(func.func):
                out = await func(*args, **kwargs_dict)
            else:
                out = func(*args, **kwargs_dict)
            return out

        _openai_fn_tool = asdict(openai_function_tool(func.func, **kwargs))
        _openai_fn_tool.pop("on_invoke_tool")
        return FunctionTool(
            native_interface=func.native_interface,
            task=func,
            report=func.report,
            on_invoke_tool=_on_invoke_tool,
            **_openai_fn_tool,
        )

    # regular callables or flyte.trace-decorated functions should use the
    # openai-agents function_tool decorator
    return openai_function_tool(func, **kwargs)
