"""CodeModeAgent — LLM + Monty sandbox orchestration with automatic retry.

The agent auto-generates its system prompt from the tool registry so that
adding a new tool is the only step required. Tools can be flyte tasks,
``@flyte.trace`` functions, or plain Python callables.

Skills (additional context injected into the system prompt) can be literal
strings or ``pathlib.Path`` objects pointing to local files.
"""

from __future__ import annotations

import inspect
import pathlib
import re
import textwrap
from typing import Any, Callable, Sequence, cast

import flyte
import flyte.sandbox

from .protocol import AgentResult

# ------------------------------------------------------------------
# Default LLM callback (litellm)
# ------------------------------------------------------------------


async def _default_call_llm(
    model: str,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    """Send a chat-completion request via *litellm* and return the text."""
    try:
        from litellm import acompletion
    except ImportError:
        raise ImportError("litellm is not installed. Please install it with `pip install litellm`.")

    all_messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        *messages,
    ]
    response = await acompletion(model=model, messages=all_messages)
    return response.choices[0].message.content  # type: ignore[union-attr]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_code(text: str) -> str:
    """Pull Python code out of markdown fences, or return the raw text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _tool_registry_key(obj: Any) -> str:
    """Stable sandbox tool name (matches :func:`flyte.sandbox.orchestrate_local` / ``_tasks_to_dict``)."""
    from flyte._task import TaskTemplate
    from flyte.remote._task import LazyEntity

    if isinstance(obj, TaskTemplate):
        # TaskTemplate itself doesn't type-expose `.func`, but AsyncFunctionTaskTemplate does.
        func = cast(Any, obj).func
        return func.__name__
    if isinstance(obj, LazyEntity):
        return obj.name.rsplit("/", maxsplit=1)[-1]
    if callable(obj):
        name = getattr(obj, "__name__", None)
        if name:
            return name
    raise TypeError(f"Cannot derive a sandbox tool name from {type(obj).__name__!r}")


def _underlying_fn_for_prompt(obj: Any) -> Callable[..., Any]:
    """Return the user function used for signatures and docstrings in prompts.

    ``TaskTemplate`` wrappers expose a generic ``(*args, **kwargs)`` signature;
    we introspect ``.func`` instead. Remote lazy task references have no local
    signature — we surface a small placeholder callable.
    """
    from flyte._task import TaskTemplate
    from flyte.remote._task import LazyEntity

    if isinstance(obj, TaskTemplate):
        # TaskTemplate itself doesn't type-expose `.func`, but AsyncFunctionTaskTemplate does.
        return cast(Any, obj).func
    if isinstance(obj, LazyEntity):

        async def _remote_task_stub(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LazyEntity tasks are resolved only inside a Flyte run.")

        ident = obj.name.rsplit("/", maxsplit=1)[-1]
        _remote_task_stub.__name__ = ident
        _remote_task_stub.__doc__ = (
            f"Remote Flyte task `{obj.name}` (lazy reference). "
            "Calls execute on the control plane when invoked from the sandbox."
        )
        return _remote_task_stub
    if callable(obj):
        return obj
    raise TypeError(f"Expected a callable or task reference, got {type(obj).__name__!r}")


def _registry_contains_task_template(reg: dict[str, Any]) -> bool:
    from flyte._task import TaskTemplate
    from flyte.remote._task import LazyEntity

    return any(isinstance(v, (TaskTemplate, LazyEntity)) for v in reg.values())


def _resolve_tools(
    tools: Sequence[Any] | dict[str, Any],
) -> dict[str, Any]:
    """Normalise *tools* into a ``{name: callable-or-task}`` dict.

    Accepts either a dict (shallow-copied) or a sequence of callables / task
    templates. Sequence entries use :func:`_tool_registry_key` for names
    (``TaskTemplate.func.__name__``, not ``str(task)``).
    """
    if isinstance(tools, dict):
        return dict(tools)
    result: dict[str, Any] = {}
    for fn in tools:
        name = _tool_registry_key(fn)
        if name in result:
            raise ValueError(f"Duplicate tool name '{name}'")
        result[name] = fn
    return result


def _load_skills(skills: Sequence[str | pathlib.Path]) -> str:
    """Concatenate skill strings / file contents into a single block."""
    parts: list[str] = []
    for skill in skills:
        if isinstance(skill, pathlib.Path):
            parts.append(skill.read_text())
        else:
            parts.append(skill)
    return "\n\n".join(parts)


# ------------------------------------------------------------------
# Traced LLM + code extraction
# ------------------------------------------------------------------


@flyte.trace
async def generate_code(
    call_llm: Callable[..., Any],
    model: str,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    """Call the LLM to generate analysis code and extract it."""
    raw = await call_llm(model, system, messages)
    return _extract_code(raw)


# ------------------------------------------------------------------
# CodeModeAgent
# ------------------------------------------------------------------


class CodeModeAgent:
    """Generates code via an LLM, executes it in a Monty sandbox, and
    optionally retries on failure.

    Parameters
    ----------
    tools:
        The callables available inside the sandbox. Accepts either a
        ``dict[str, Callable]`` mapping or a sequence containing any mix of:

        - plain Python functions
        - ``@flyte.trace`` helpers
        - ``@env.task`` :class:`~flyte.TaskTemplate` instances (durable)
        - :class:`~flyte.remote._task.LazyEntity` references to remote tasks

        Sequence entries are keyed by ``TaskTemplate.func.__name__`` for tasks
        (and ``__name__`` for plain callables). Pass a dict to expose a tool
        under a different name to the LLM (e.g.
        ``tools={"fetch_data": durable_fetch_with_retries}``). The sandbox
        receives the original objects, so ``@env.task`` entries execute
        durably on the cluster; the prompt introspects ``TaskTemplate.func``
        to surface the underlying signature and docstring.
    model:
        Model identifier passed to *call_llm*.
    max_retries:
        How many *additional* attempts after the first failure.
    skills:
        Extra context injected into the system prompt. Each entry is
        either a literal string or a ``pathlib.Path`` to a local file
        whose contents will be read.
    call_llm:
        Async callback ``(model, system, messages) -> str``. Defaults to
        a *litellm*-based implementation.
    system_prompt_prefix:
        Optional text prepended to the auto-generated system prompt,
        allowing callers to set the agent persona or additional
        instructions.
    """

    def __init__(
        self,
        tools: Sequence[Callable] | dict[str, Callable],
        *,
        model: str = "claude-haiku-4-5",
        max_retries: int = 2,
        skills: Sequence[str | pathlib.Path] = (),
        call_llm: Callable[..., Any] = _default_call_llm,
        system_prompt_prefix: str | None = None,
    ) -> None:
        self._tools = _resolve_tools(tools)
        self._model = model
        self._max_retries = max_retries
        self._skills = skills
        self._call_llm = call_llm
        self._system_prompt_prefix = system_prompt_prefix
        self.system_prompt = self._build_system_prompt()

    def uses_flyte_tools(self) -> bool:
        """True when the tool registry contains Flyte tasks or remote lazy tasks."""
        return _registry_contains_task_template(self._tools)

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        tool_lines: list[str] = []
        for name, fn in self._tools.items():
            u = _underlying_fn_for_prompt(fn)
            sig = inspect.signature(u)
            doc = inspect.getdoc(u) or ""
            indented_doc = textwrap.indent(doc, "        ")
            tool_lines.append(f"    - {name}{sig}\n{indented_doc}")

        tools_block = "\n\n".join(tool_lines)

        restrictions = flyte.sandbox.ORCHESTRATOR_SYNTAX_PROMPT.replace("{", "{{").replace("}", "}}")

        prefix = self._system_prompt_prefix or (
            "You are a helpful assistant. Write Python code to accomplish the user's request."
        )

        skills_section = ""
        if self._skills:
            skills_text = _load_skills(self._skills)
            skills_section = f"\n\nAdditional skills / context:\n{skills_text}"

        return (
            textwrap.dedent("""\
            {prefix}

            Available functions:
        {tools}

            {restrictions}
            - Return a dict with relevant result keys.{skills}
        """)
            .replace("{prefix}", prefix)
            .replace("{tools}", tools_block)
            .replace("{restrictions}", restrictions)
            .replace("{skills}", skills_section)
        )

    # ------------------------------------------------------------------
    # Tool descriptions (for UI sidebars, etc.)
    # ------------------------------------------------------------------

    def tool_descriptions(self) -> list[dict[str, str]]:
        """Return JSON-friendly metadata for every registered tool."""
        descs: list[dict[str, str]] = []
        for name, fn in self._tools.items():
            u = _underlying_fn_for_prompt(fn)
            sig = f"{name}{inspect.signature(u)}"
            doc = inspect.getdoc(u) or ""
            short_doc = doc.split("\n\n")[0].replace("\n", " ").strip()
            descs.append({"name": name, "signature": sig, "description": short_doc})
        return descs

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    async def _execute(self, code: str) -> Any:
        """Run *code* in a Monty sandbox with the registered tools."""
        return await flyte.sandbox.orchestrate_local(
            code,
            inputs={"_unused": 0},
            tasks=list(self._tools.values()),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        """Generate code, execute in sandbox, retry on failure."""
        messages: list[dict[str, str]] = [*history, {"role": "user", "content": message}]

        try:
            code = await generate_code(self._call_llm, self._model, self.system_prompt, messages)
        except Exception as exc:
            return AgentResult(error=f"Code generation failed: {exc}")

        attempts = 1

        for attempt in range(1 + self._max_retries):
            attempts = attempt + 1
            with flyte.group(f"codemode-attempt-{attempt}"):
                try:
                    result = await self._execute(code)
                except Exception as exc:
                    if attempt < self._max_retries:
                        retry_content = (
                            f"Your previous code failed with this error:\n\n"
                            f"```\n{exc}\n```\n\n"
                            f"The code that failed:\n\n"
                            f"```python\n{code}\n```\n\n"
                            f"Please fix the code. Remember the Monty sandbox restrictions."
                        )
                        retry_messages = [
                            *messages,
                            {"role": "assistant", "content": f"```python\n{code}\n```"},
                            {"role": "user", "content": retry_content},
                        ]
                        try:
                            code = await generate_code(
                                self._call_llm,
                                self._model,
                                self.system_prompt,
                                retry_messages,
                            )
                        except Exception as llm_exc:
                            return AgentResult(
                                code=code,
                                error=f"Retry LLM call failed: {llm_exc}",
                                attempts=attempts,
                            )
                        continue
                    return AgentResult(
                        code=code,
                        error=f"Sandbox execution failed after {attempts} attempt(s): {exc}",
                        attempts=attempts,
                    )

            charts = result.get("charts", []) if isinstance(result, dict) else []
            summary = result.get("summary", "No summary generated.") if isinstance(result, dict) else str(result)
            return AgentResult(code=code, charts=charts, summary=summary, attempts=attempts)

        return AgentResult(code=code, error="Unexpected: exhausted retries", attempts=attempts)
