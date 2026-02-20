"""CodeModeAgent — LLM + sandbox orchestration with automatic retry.

The agent auto-generates its system prompt from the tool registry so that
adding a new tool to ``_tools.ALL_TOOLS`` is the only step required.
"""

from __future__ import annotations

import inspect
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import flyte
import flyte.sandbox

# ------------------------------------------------------------------
# LLM call + code extraction (module-level for @flyte.trace compat)
# ------------------------------------------------------------------


async def _call_llm(
    model: str,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    """Send a chat-completion request and return the text response."""
    import httpx

    api_key = os.environ["ANTHROPIC_API_KEY"]
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "system": system,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data["content"][0]["text"]  # type: ignore[no-any-return]


def _extract_code(text: str) -> str:
    """Pull Python code out of markdown fences, or return the raw text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


@flyte.trace
async def generate_code(
    model: str,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    """Call Claude to generate analysis code and extract it."""
    raw = await _call_llm(model, system, messages)
    return _extract_code(raw)


@dataclass
class AgentResult:
    """Outcome of a single ``CodeModeAgent.run`` invocation."""

    code: str = ""
    charts: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    attempts: int = 1


class CodeModeAgent:
    """Generates analysis code via an LLM, executes it in a Monty sandbox,
    and optionally retries on failure.

    Parameters
    ----------
    tools:
        Mapping of tool-name -> callable.  Signatures and docstrings are
        introspected to build the system prompt automatically.
    execution_tools:
        Optional mapping of tool-name -> callable used at *execution* time
        in the sandbox.  When ``None`` (the default), ``tools`` is used for
        both prompt generation and execution.  Pass ``@env.task``-wrapped
        versions here for durable execution through the controller.
    model:
        Anthropic model ID.
    max_retries:
        How many *additional* attempts after the first failure (so
        ``max_retries=2`` means up to 3 total attempts).
    """

    def __init__(
        self,
        tools: dict[str, Callable],
        *,
        execution_tools: dict[str, Callable] | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 2,
    ) -> None:
        self._tools = tools  # for prompt generation
        self._execution_tools = execution_tools or tools  # for sandbox
        self._model = model
        self._max_retries = max_retries
        self.system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        tool_lines: list[str] = []
        for name, fn in self._tools.items():
            sig = inspect.signature(fn)
            doc = inspect.getdoc(fn) or ""
            # Indent the docstring body under the signature
            indented_doc = textwrap.indent(doc, "        ")
            tool_lines.append(f"    - {name}{sig}\n{indented_doc}")

        tools_block = "\n\n".join(tool_lines)

        return textwrap.dedent("""\
            You are a data analyst. Write Python code to analyze data and produce charts.

            Available functions:
        {tools}

            CRITICAL — Sandbox syntax restrictions (Monty runtime):
            - No imports.
            - No subscript assignment: `d[key] = value` and `l[i] = value` are FORBIDDEN.
            - Reading subscripts is OK: `x = d[key]` and `x = l[i]` work fine.
            - Build lists with .append() and list literals, NOT by index assignment.
            - Build dicts ONLY as literals: {{"k": v, ...}}. Never mutate them after creation.
            - To aggregate data, use lists of tuples/dicts, not mutating a dict.
            - The last expression in your code must be the return value.
            - Return a dict: {{"charts": [<html strings from create_chart>], "summary": "<text>"}}

            Example — group sales by region (correct pattern):
                data = fetch_data("sales_2024")
                months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                regions = ["North", "South", "East", "West"]

                # Build per-region series using list comprehensions (NO dict mutation)
                series = []
                for region in regions:
                    region_data = [row["revenue"] for row in data if row["region"] == region]
                    series.append({{"label": region, "data": region_data}})

                chart1 = create_chart("line", "Revenue by Region", months, series)

                total = 0
                for row in data:
                    total = total + row["revenue"]

                {{"charts": [chart1], "summary": "Total 2024 revenue: $" + str(total)}}
        """).replace("{tools}", tools_block)

    # ------------------------------------------------------------------
    # Tool descriptions for the /api/tools sidebar
    # ------------------------------------------------------------------

    def tool_descriptions(self) -> list[dict[str, str]]:
        """Return JSON-friendly metadata for every registered tool."""
        descs: list[dict[str, str]] = []
        for name, fn in self._tools.items():
            sig = f"{name}{inspect.signature(fn)}"
            doc = inspect.getdoc(fn) or ""
            # Use only the first paragraph of the docstring
            short_doc = doc.split("\n\n")[0].replace("\n", " ").strip()
            descs.append({"name": name, "signature": sig, "description": short_doc})
        return descs

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    async def _execute(self, code: str) -> dict[str, Any]:
        """Run *code* in a Monty sandbox with the registered tools."""
        result = await flyte.sandbox.orchestrate_local(
            code,
            inputs={"_unused": 0},
            tasks=list(self._execution_tools.values()),
        )
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        """Generate code, execute in sandbox, retry on failure.

        Returns an ``AgentResult`` with code, charts, summary, error, and
        the number of attempts made.
        """
        messages: list[dict[str, str]] = [*history, {"role": "user", "content": message}]

        # First attempt: generate code
        try:
            code = await generate_code(self._model, self.system_prompt, messages)
        except Exception as exc:
            return AgentResult(error=f"Code generation failed: {exc}")

        attempts = 1

        # Execute + retry loop
        for attempt in range(1 + self._max_retries):
            attempts = attempt + 1
            try:
                result = await self._execute(code)
            except Exception as exc:
                if attempt < self._max_retries:
                    # Ask the LLM to fix its own code
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
                        code = await generate_code(self._model, self.system_prompt, retry_messages)
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

            # Success — extract charts + summary
            charts = result.get("charts", []) if isinstance(result, dict) else []
            summary = result.get("summary", "No summary generated.") if isinstance(result, dict) else str(result)
            return AgentResult(code=code, charts=charts, summary=summary, attempts=attempts)

        # Should not be reached, but just in case:
        return AgentResult(code=code, error="Unexpected: exhausted retries", attempts=attempts)
