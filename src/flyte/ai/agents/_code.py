"""Code-mode helpers for :class:`flyte.ai.agents.Agent`.

In *code mode* the agent does not emit JSON tool calls. Instead, on each turn the
LLM writes a short Python program that is executed in the Monty sandbox
(``flyte.sandbox.orchestrate_local``). The agent's tools are exposed to that
sandbox as plain functions, so ``@env.task`` tools dispatch durably on the
cluster, ``@flyte.trace`` helpers are traced, and ``flyte_map(...)`` fans out in
parallel — all the usual Flyte features, driven by generated code.

This module is internal: the public surface is the ``code_mode`` flag on
:class:`~flyte.ai.agents.Agent`.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Any

from ._tools import AgentTool, _summarize_signature

# A python code fence (```python / ```py / bare ```). Captures the body.
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def extract_python_code(text: str | None) -> str | None:
    """Return the first fenced Python block in *text*, or ``None`` if absent.

    The multi-turn code-mode loop uses the *presence* of a code block to decide
    whether to keep iterating: a turn with a
    code block is executed, while a turn without one is treated as the final
    answer. So this returns ``None`` (rather than the whole text) when no fence is
    found.
    """
    if not text:
        return None
    match = _CODE_FENCE_RE.search(text)
    if not match:
        return None
    code = match.group(1).strip()
    return code or None


def _underlying_callable(tool: AgentTool) -> Any:
    """The function used for signature/docstring introspection in the prompt."""
    from flyte._task import TaskTemplate

    target = tool.target
    if isinstance(target, TaskTemplate):
        return getattr(target, "func", None)
    if callable(target):
        # ``@flyte.trace`` wraps the user function; introspect the original.
        return getattr(target, "__wrapped__", target)
    return None


def _sandbox_name(tool: AgentTool) -> str:
    """The name the sandbox exposes a tool under.

    Mirrors ``flyte.sandbox`` name derivation (``TaskTemplate.func.__name__`` /
    ``callable.__name__``) so the generated code calls match the prompt. Tools
    without an introspectable target (MCP / hand-built ``AgentTool``) fall back
    to the tool's LLM-facing name.
    """
    from flyte._task import TaskTemplate

    target = tool.target
    if isinstance(target, TaskTemplate):
        return getattr(target, "func").__name__
    if callable(target):
        name = getattr(target, "__name__", None)
        if name:
            return name
    return tool.name


def _make_execute_wrapper(tool: AgentTool) -> Any:
    """Wrap an ``AgentTool`` with no introspectable target as a sandbox callable."""

    async def _wrapper(**kwargs: Any) -> Any:
        return await tool.execute(kwargs)

    _wrapper.__name__ = tool.name
    _wrapper.__doc__ = tool.description
    return _wrapper


def build_sandbox_tools(registry: dict[str, AgentTool]) -> list[Any]:
    """Map the agent's tool registry to objects ``orchestrate_local`` understands.

    ``@env.task`` / ``LazyEntity`` / plain-callable tools are passed through as
    their underlying object so they keep their native dispatch (durable tasks
    run on-cluster). Tools without such a target (e.g. MCP or hand-built
    :class:`AgentTool`) are wrapped in a named async shim over ``execute``.

    Raises:
        ValueError: if two tools resolve to the same sandbox function name. The
            sandbox keys functions by name, so a collision would make one tool
            unreachable; we fail fast here with a clear message rather than
            letting it surface as a confusing runtime error inside the loop.
    """
    from flyte._task import TaskTemplate

    out: list[Any] = []
    seen: dict[str, str] = {}
    for tool in registry.values():
        name = _sandbox_name(tool)
        if name in seen:
            raise ValueError(
                f"Code mode requires distinct sandbox function names, but tools {seen[name]!r} and "
                f"{tool.name!r} both resolve to {name!r}. Rename one of them (e.g. via "
                "tool(..., name=...) or by renaming the underlying function)."
            )
        seen[name] = tool.name

        target = tool.target
        if isinstance(target, TaskTemplate) or (callable(target) and getattr(target, "__name__", None)):
            out.append(target)
        else:
            out.append(_make_execute_wrapper(tool))
    return out


def _tool_prompt_block(tool: AgentTool) -> str:
    """A single function entry for the code-mode system prompt."""
    underlying = _underlying_callable(tool)
    name = _sandbox_name(tool)
    if underlying is not None:
        try:
            signature = str(inspect.signature(underlying))
        except (TypeError, ValueError):
            signature = "(...)"
        doc = inspect.getdoc(underlying) or tool.description or ""
        header = f"{name}{signature}"
    else:
        # No introspectable target — synthesize a readable pseudo-signature.
        header = _summarize_signature(tool)
        doc = tool.description or ""

    if doc:
        return f"    - {header}\n{textwrap.indent(doc, '        ')}"
    return f"    - {header}"


def build_code_system_prompt(
    instructions: str,
    registry: dict[str, AgentTool],
    skills_block: str = "",
) -> str:
    """Build the system prompt that drives the code-generation loop."""
    import flyte.sandbox

    tool_lines = [_tool_prompt_block(tool) for tool in registry.values()]
    tools_block = "\n\n".join(tool_lines) if tool_lines else "    (no functions available)"
    restrictions = flyte.sandbox.ORCHESTRATOR_SYNTAX_PROMPT

    return (
        f"{instructions}\n\n"
        "You operate in CODE MODE. You make progress by writing small Python programs that run "
        "in a sandbox; the available tools are exposed as ordinary Python functions.\n\n"
        "On each turn, do exactly one of the following:\n"
        "1. Write a SINGLE fenced Python code block (```python ... ```) that calls the functions "
        "below to take the next step. The code runs in a Monty sandbox and the value of its LAST "
        "expression is returned to you as the observation for the next turn.\n"
        "2. When you have the final answer and no more code needs to run, reply with plain text and "
        "NO code block. That plain-text reply is your final answer to the user.\n\n"
        "Available functions (call them directly inside your code):\n"
        f"{tools_block}\n\n"
        f"{restrictions}\n\n"
        "Guidance:\n"
        "- Take one focused step per code block and inspect the returned observation before continuing.\n"
        "- End each code block with the value(s) you want to observe (the last expression is returned).\n"
        "- Do not wrap results in custom classes; return plain dicts/lists/tuples and primitives.\n"
        "- When finished, summarize the outcome for the user in plain text (no code block)."
        f"{skills_block}"
    )
