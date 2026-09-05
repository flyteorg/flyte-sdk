"""CodeModeAgent — LLM + sandbox orchestration with automatic retry.

The agent auto-generates its system prompt from the tool registry so that
adding a new tool to ``_tools.ALL_TOOLS`` is the only step required.

Two entry points:

- ``run_streaming`` — async generator yielding one event per phase
  (``llm_start``, ``llm_done``, ``exec_start``, ``retry``, ``done``) so a UI
  can show live progress and per-phase timings.
- ``run`` — convenience wrapper that drains the stream and returns the final
  ``AgentResult``.
"""

from __future__ import annotations

import inspect
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import flyte
import flyte.sandbox
from flyte.syncify import syncify

DEFAULT_MODEL = os.environ.get("CODEMODE_MODEL", "claude-sonnet-4-6")

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


def _numbered(code: str) -> str:
    return "\n".join(f"{i + 1:3d} | {line}" for i, line in enumerate(code.splitlines()))


@dataclass
class AgentResult:
    """Outcome of a single ``CodeModeAgent.run`` invocation."""

    code: str = ""
    charts: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    attempts: int = 1
    llm_duration_s: float = 0.0
    execution_duration_s: float = 0.0


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
        model: str = DEFAULT_MODEL,
        max_retries: int = 2,
    ) -> None:
        self._tools = tools  # for prompt generation
        self._execution_tools = execution_tools or tools  # for sandbox
        self._model = model
        self._max_retries = max_retries
        self.system_prompt = self._build_system_prompt()

    @property
    def model(self) -> str:
        """The LLM model ID used for code generation."""
        return self._model

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

        # Escape braces for .format() — the ORCHESTRATOR_SYNTAX_PROMPT uses
        # literal braces (e.g. dict examples) that must survive formatting.
        restrictions = flyte.sandbox.ORCHESTRATOR_SYNTAX_PROMPT.replace("{", "{{").replace("}", "}}")

        return (
            textwrap.dedent("""\
            You are a data analyst. Write Python code to analyze data and produce charts and tables.

            ALWAYS respond with Python code, for EVERY message, no exceptions. For greetings or
            questions you cannot answer with data, return code whose result dict carries a helpful
            message in "summary" and an empty "charts" list. Never reply with plain prose.

            Available functions:
        {tools}

            {restrictions}
            - Return a dict: {{"charts": [<html strings from create_chart / create_table>], "summary": "<text>"}}
            - Build the result values first; the final dict literal must be the LAST line.

            When to use tables vs charts:
            - create_table() for listings, rankings and multi-column breakdowns read row by row.
            - create_chart() for trends over time, comparisons of one metric, and proportions.
            - Combine both when useful (e.g. a chart plus a table of exact values).

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

                totals = group_and_aggregate(data, "region", "revenue", "sum")
                table_rows = [[t["group"], "$" + str(t["value"])] for t in totals]
                table1 = create_table("Revenue by Region", ["Region", "Revenue"], table_rows)

                total = 0
                for row in data:
                    total = total + row["revenue"]

                {{"charts": [chart1, table1], "summary": "Total 2024 revenue: $" + str(total)}}
        """)
            .replace("{tools}", tools_block)
            .replace("{restrictions}", restrictions)
        )

    # ------------------------------------------------------------------
    # Tool descriptions for the /api/tools sidebar
    # ------------------------------------------------------------------

    def tool_descriptions(self) -> list[dict[str, str]]:
        """Return JSON-friendly metadata for every registered tool."""
        descs: list[dict[str, str]] = []
        for name, fn in self._tools.items():
            sig = f"{name}{inspect.signature(fn)}"
            doc = inspect.getdoc(fn) or ""
            descs.append({"name": name, "signature": sig, "description": doc})
        return descs

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    async def _execute(self, code: str) -> Any:
        """Run *code* in a Monty sandbox with the registered tools."""
        return await flyte.sandbox.orchestrate_local(
            code,
            inputs={},
            tasks=list(self._execution_tools.values()),
        )

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    async def run_streaming(self, message: str, history: list[dict[str, str]]) -> AsyncIterator[dict[str, Any]]:
        """Generate code, execute in sandbox, retry on failure — yielding phase events.

        Events (all JSON-serialisable dicts):
            {"phase": "llm_start", "attempt": n}
            {"phase": "llm_done", "attempt": n, "llm_duration_s": s}
            {"phase": "exec_start", "attempt": n}
            {"phase": "retry", "attempt": n, "error": "..."}
            {"phase": "done", code, charts, summary, error, attempts,
                              llm_duration_s, execution_duration_s}
        """
        messages: list[dict[str, str]] = [*history, {"role": "user", "content": message}]
        code = ""
        attempts = 0
        total_llm_s = 0.0
        total_exec_s = 0.0

        def done(**fields: Any) -> dict[str, Any]:
            return {
                "phase": "done",
                "code": code,
                "charts": [],
                "summary": "",
                "error": "",
                "attempts": attempts,
                "llm_duration_s": round(total_llm_s, 2),
                "execution_duration_s": round(total_exec_s, 2),
                **fields,
            }

        for attempt in range(1 + self._max_retries):
            attempts = attempt + 1

            # --- LLM phase ---
            yield {"phase": "llm_start", "attempt": attempts}
            t0 = time.monotonic()
            try:
                code = await generate_code(self._model, self.system_prompt, messages)
            except Exception as exc:
                yield done(error=f"Code generation failed: {exc}")
                return
            llm_s = time.monotonic() - t0
            total_llm_s += llm_s
            yield {"phase": "llm_done", "attempt": attempts, "llm_duration_s": round(llm_s, 2)}

            # --- Execution phase ---
            yield {"phase": "exec_start", "attempt": attempts}
            t1 = time.monotonic()
            try:
                result = await self._execute(code)
            except Exception as exc:
                total_exec_s += time.monotonic() - t1
                if attempt < self._max_retries:
                    yield {"phase": "retry", "attempt": attempts, "error": str(exc)}
                    # Ask the LLM to fix its own code
                    messages = [
                        *messages,
                        {"role": "assistant", "content": f"```python\n{code}\n```"},
                        {
                            "role": "user",
                            "content": (
                                f"Your previous code failed with this error:\n\n```\n{exc}\n```\n\n"
                                f"The code that failed (with line numbers):\n\n```\n{_numbered(code)}\n```\n\n"
                                "Please fix the code. Remember the Monty sandbox restrictions."
                            ),
                        },
                    ]
                    continue
                yield done(error=f"Sandbox execution failed after {attempts} attempt(s): {exc}")
                return
            total_exec_s += time.monotonic() - t1

            # Success — extract charts + summary
            charts = result.get("charts", []) if isinstance(result, dict) else []
            summary = result.get("summary", "No summary generated.") if isinstance(result, dict) else str(result)
            yield done(charts=charts, summary=summary)
            return

        yield done(error="Unexpected: exhausted retries")

    @syncify
    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        """Non-streaming variant: drain ``run_streaming`` and return the final result."""
        final: dict[str, Any] = {}
        async for event in self.run_streaming(message, history):
            if event["phase"] == "done":
                final = event
        return AgentResult(
            code=final.get("code", ""),
            charts=final.get("charts", []),
            summary=final.get("summary", ""),
            error=final.get("error", ""),
            attempts=final.get("attempts", 1),
            llm_duration_s=final.get("llm_duration_s", 0.0),
            execution_duration_s=final.get("execution_duration_s", 0.0),
        )
