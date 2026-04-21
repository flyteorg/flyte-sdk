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
from typing import Any, Callable, Sequence

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
    from litellm import acompletion

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


def _resolve_tools(
    tools: Sequence[Callable] | dict[str, Callable],
) -> dict[str, Callable]:
    """Normalise *tools* into a ``{name: callable}`` dict.

    Accepts either a dict (returned as-is) or a sequence of callables whose
    ``__name__`` attribute is used as the key.
    """
    if isinstance(tools, dict):
        return dict(tools)
    result: dict[str, Callable] = {}
    for fn in tools:
        name = getattr(fn, "__name__", None) or str(fn)
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
        ``dict[str, Callable]`` mapping or a sequence of callables (whose
        ``__name__`` becomes the key). These can be flyte tasks,
        ``@flyte.trace`` functions, or plain Python functions.
    execution_tools:
        Optional mapping used at *execution* time in the sandbox. When
        ``None`` (the default), *tools* is used for both prompt generation
        and execution.
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
        execution_tools: Sequence[Callable] | dict[str, Callable] | None = None,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 2,
        skills: Sequence[str | pathlib.Path] = (),
        call_llm: Callable[..., Any] = _default_call_llm,
        system_prompt_prefix: str | None = None,
    ) -> None:
        self._tools = _resolve_tools(tools)
        self._execution_tools = _resolve_tools(execution_tools) if execution_tools else self._tools
        self._model = model
        self._max_retries = max_retries
        self._skills = skills
        self._call_llm = call_llm
        self._system_prompt_prefix = system_prompt_prefix
        self.system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        tool_lines: list[str] = []
        for name, fn in self._tools.items():
            sig = inspect.signature(fn)
            doc = inspect.getdoc(fn) or ""
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
            sig = f"{name}{inspect.signature(fn)}"
            doc = inspect.getdoc(fn) or ""
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
            tasks=list(self._execution_tools.values()),
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
