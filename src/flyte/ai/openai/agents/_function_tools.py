import typing

from flyte._task import AsyncFunctionTaskTemplate
from flyte._utils import lazy_module


if typing.TYPE_CHECKING:
    import agents
    from agents import function_tool as openai_function_tool, FunctionTool
else:
    agents = lazy_module("agents")


def function_tool(
    func: AsyncFunctionTaskTemplate,
) -> agents.FunctionTool:
    import ipdb; ipdb.set_trace()
    return agents.FunctionTool(
        name=func.name,
        description=func.description,
        params_json_schema=func.params_json_schema,
        on_invoke_tool=func.on_invoke_tool,
    )
