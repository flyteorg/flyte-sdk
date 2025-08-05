"""Unit tests for OpenAI agents package."""

import flyte
from flyte.ai.openai.agents import function_tool
from flyte.ai.openai.agents._function_tools import FunctionTool
from agents import FunctionTool as OpenAIFunctionTool


def test_task():
    env = flyte.TaskEnvironment("foo")

    @env.task
    def my_task(prompt: str) -> str:
        return f"Hello, {prompt}!"
    
    tool = function_tool(my_task)
    assert isinstance(tool, FunctionTool)
    assert tool.task is my_task
    assert tool.report == my_task.report
    assert tool.native_interface is my_task.native_interface
    assert hasattr(tool, "execute")


def test_trace():
    
    @flyte.trace
    def my_function(prompt: str) -> str:
        return f"Hello, {prompt}!"
    
    tool = function_tool(my_function)
    assert isinstance(tool, OpenAIFunctionTool)


def test_function():

    def my_function(prompt: str) -> str:
        return f"Hello, {prompt}!"
    
    tool = function_tool(my_function)
    assert isinstance(tool, OpenAIFunctionTool)
