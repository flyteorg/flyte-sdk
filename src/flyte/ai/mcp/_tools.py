"""MCP tool implementations for `flyte.ai.mcp.FlyteMCPAppEnvironment`.

The tools live here rather than in ``_flyte_mcp_app.py`` purely for size: the environment
module owns configuration, the tool registry and registration, this one owns the bodies.
Every tool closes over the environment (``env``) for its allowlists, search paths and the
per-call project/domain scoping, so they are built by `build_tool_functions` rather
than defined at module level.

Each function's docstring is what the MCP client shows the model, so they are written for an
agent audience: when to reach for the tool, and what it hands back.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from datetime import timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, cast

import flyte
import flyte.remote

if TYPE_CHECKING:
    from flyte.ai.mcp._flyte_mcp_app import FlyteMCPAppEnvironment

#: Hard ceiling on ``get_logs(max_lines=...)`` so a chatty action cannot blow the context window.
MAX_LOG_LINES = 2000

#: Overall budget for collecting log lines before returning what we have.
LOG_COLLECT_TIMEOUT_S = 30.0

#: How many server-side entries the ``list_*`` tools scan when an allowlist is configured.
#: The allowlist is applied client-side, so passing the caller's ``limit`` straight through would
#: let the server truncate before filtering and hide allowed entries that sit past the first page.
ALLOWLIST_SCAN_LIMIT = 1000

#: Wall-clock budget for one ``search_*`` call. The pattern is caller-supplied, so a
#: catastrophically backtracking regex must not run unbounded.
SEARCH_TIMEOUT_S = 10.0

#: Longest accepted search pattern. Bounds both compile cost and the space of pathological
#: expressions a caller can express.
MAX_SEARCH_PATTERN_LEN = 200

#: Per-line matching budget. Caps how long one backtracking match can hold the GIL, which is
#: what keeps a hostile pattern from stalling other tenants' requests.
PER_LINE_MATCH_TIMEOUT_S = 0.05

#: How many searches may occupy worker threads at once. Slow searches then queue instead of
#: exhausting the default thread pool and starving every other tool.
MAX_CONCURRENT_SEARCHES = 4

#: Constructed at import: since 3.10 a Semaphore binds to the running loop on first use, not
#: at construction, so this is safe at module scope.
_SEARCH_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)


# NOTE: This module uses `from __future__ import annotations`, which means annotations on the
# tool functions are stored as strings. FastMCP evaluates those strings against this module's
# globals, so the context type has to be importable here.
try:  # pragma: no cover
    from mcp.server.fastmcp import Context as MCPContext
except ModuleNotFoundError:  # pragma: no cover

    class MCPContext:  # type: ignore[no-redef]
        pass


try:  # pragma: no cover
    from mcp.server.fastmcp.exceptions import ToolError
except ModuleNotFoundError:  # pragma: no cover

    class ToolError(Exception):  # type: ignore[no-redef]
        pass


# ------------------------------
# Allowlist helpers
# ------------------------------


def _is_task_allowed(allowlist: list[str] | None, domain: str, project: str, name: str) -> bool:
    if allowlist is None:
        return True

    full_path = f"{domain}/{project}/{name}"
    for allowed in allowlist:
        if allowed == full_path:
            return True
        if "/" not in allowed and allowed == name:
            return True
        if allowed.count("/") == 1 and allowed == f"{project}/{name}":
            return True
    return False


def _is_app_allowed(allowlist: list[str] | None, name: str) -> bool:
    if allowlist is None:
        return True
    return name in allowlist


def _is_trigger_allowed(allowlist: list[str] | None, task_name: str, trigger_name: str) -> bool:
    if allowlist is None:
        return True

    full_path = f"{task_name}/{trigger_name}"
    for allowed in allowlist:
        if allowed == full_path:
            return True
        if "/" not in allowed and allowed == trigger_name:
            return True
    return False


# ------------------------------
# Serialization helpers
# ------------------------------


def _phase_str(phase: Any) -> str | None:
    """Render an ``ActionPhase`` (a ``str`` enum) as a plain JSON-able string."""
    if phase is None:
        return None
    return getattr(phase, "value", None) or str(phase)


def _ts(message: Any, field_name: str) -> str | None:
    """ISO-8601 rendering of a protobuf ``Timestamp`` field, or None when unset."""
    try:
        if not message.HasField(field_name):
            return None
    except ValueError:  # pragma: no cover - field not present in this proto build
        return None
    return getattr(message, field_name).ToDatetime().replace(tzinfo=timezone.utc).isoformat()


def _seconds(delta: timedelta | None) -> float | None:
    return None if delta is None else round(delta.total_seconds(), 3)


def _type_name(tpe: Any) -> str:
    """Render a Python type the way a caller would write it, for the ``get_task`` interface map."""
    if isinstance(tpe, str):
        return tpe
    if isinstance(tpe, type) and not hasattr(tpe, "__origin__"):
        return tpe.__name__
    return repr(tpe)


def _pb_to_dict(message: Any) -> dict | None:
    """Convert a protobuf message to a plain dict; ``None`` passes through."""
    if message is None:
        return None
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(message, preserving_proto_field_name=True)


# ------------------------------
# Search helper
# ------------------------------


def _line_matcher(pattern: str) -> tuple[Callable[[str], bool], str]:
    """Return a per-line predicate for ``pattern`` plus a note about how it will match.

    Patterns are regexes, but they come from the caller, and ``re`` cannot bound backtracking:
    ``(a|a)+b`` against a 40-character line is a single C call that pins the GIL for minutes, so
    it would stall every other tenant's request, not just this one. Matching therefore goes
    through ``regex``, which accepts a per-match timeout. Without that package installed the
    predicate degrades to a literal substring search, which cannot backtrack at all.

    An invalid regex is not something the caller can act on mid-conversation, so it also
    degrades to a substring search. Either fallback is reported in the search output.
    """
    try:
        import regex
    except ModuleNotFoundError:
        return (lambda line: pattern in line), "matched as a literal substring (install flyte[mcp] for regex search)"

    try:
        rx = regex.compile(pattern)
    except regex.error:
        return (lambda line: pattern in line), "pattern is not a valid regex; matched as a literal substring"

    def matches(line: str) -> bool:
        try:
            return rx.search(line, timeout=PER_LINE_MATCH_TIMEOUT_S) is not None
        except TimeoutError:
            # This line out-ran its budget; treat it as a non-match rather than failing the
            # whole search, and let the overall deadline stop the scan if it keeps happening.
            return False

    return matches, ""


async def _search_files(
    pattern: str,
    path: str,
    *,
    top_n: int = 3,
    before_context_lines: int = 5,
    after_context_lines: int = 5,
) -> str:
    """Search files for the regex ``pattern`` and return Markdown with excerpt blocks.

    Recursively scans up to 5000 files under ``path`` (or reads ``path`` if it is
    a file). Files are ranked by match count; the top ``top_n`` files get merged
    context windows around each hit. The first line reports how many files matched
    versus how many are shown, so a near-miss is never silent.

    The scan itself is CPU-bound and the pattern comes from the caller, so it runs in a worker
    thread under a deadline and a concurrency cap: a catastrophically backtracking regex
    (``(a|a)+b``) then costs one slow request instead of freezing the whole server for every
    other tenant.
    """
    if len(pattern) > MAX_SEARCH_PATTERN_LEN:
        return f"Error: pattern is too long (max {MAX_SEARCH_PATTERN_LEN} characters)."

    # A worker thread cannot be cancelled, so the slot is held until the thread itself finishes
    # rather than until this coroutine stops waiting. Releasing on timeout would let a stream of
    # pathological patterns pile up threads without bound.
    await _SEARCH_SLOTS.acquire()
    task = asyncio.ensure_future(
        asyncio.to_thread(
            _search_files_blocking,
            pattern,
            path,
            top_n=top_n,
            before_context_lines=before_context_lines,
            after_context_lines=after_context_lines,
        )
    )
    task.add_done_callback(lambda _: _SEARCH_SLOTS.release())

    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=SEARCH_TIMEOUT_S)
    except asyncio.TimeoutError:
        return (
            f"Error: search timed out after {SEARCH_TIMEOUT_S:.0f}s. Simplify the pattern — "
            "nested quantifiers such as `(a|a)+b` can backtrack exponentially."
        )


def _search_files_blocking(
    pattern: str,
    path: str,
    *,
    top_n: int = 3,
    before_context_lines: int = 5,
    after_context_lines: int = 5,
) -> str:
    """Synchronous body of `_search_files`; always called in a worker thread."""
    deadline = time.monotonic() + SEARCH_TIMEOUT_S
    try:
        p = pathlib.Path(path)
        if not p.exists():
            return f"Error: path does not exist: {path}"

        matches_line, match_note = _line_matcher(pattern)

        files: list[pathlib.Path] = []
        if p.is_file():
            files = [p]
        else:
            # avoid pathological scans
            for fp in p.rglob("*"):
                if fp.is_file():
                    files.append(fp)
                if len(files) >= 5000:
                    break

        ranked: list[tuple[int, pathlib.Path, list[str], list[int]]] = []
        timed_out = False
        for fp in files:
            # The awaiting coroutine gives up at the same deadline; checking here too means the
            # thread stops burning CPU instead of running on unnoticed.
            if time.monotonic() > deadline:
                timed_out = True
                break

            try:
                text = fp.read_text(errors="ignore")
            except Exception:
                continue

            if not text:
                continue
            lines = text.splitlines()
            match_idxs = [i for i, line in enumerate(lines) if matches_line(line)]
            if not match_idxs:
                continue

            ranked.append((len(match_idxs), fp, lines, match_idxs))

        if not ranked:
            if timed_out:
                return "Search timed out before any match was found; simplify the pattern."
            return "No matches found"

        ranked.sort(key=lambda t: t[0], reverse=True)
        shown = min(len(ranked), max(1, top_n))
        header = f"{len(ranked)} files matched; showing top {shown}"
        if match_note:
            header += f" ({match_note})"
        if timed_out:
            header += " (search stopped early at the time limit; results may be incomplete)"

        blocks: list[str] = [header, ""]
        for count, fp, lines, match_idxs in ranked[:shown]:
            # collect context windows around each match (merge overlapping)
            windows: list[tuple[int, int]] = []
            for i in match_idxs:
                start = max(0, i - before_context_lines)
                end = min(len(lines), i + after_context_lines + 1)
                windows.append((start, end))
            windows.sort()
            merged: list[tuple[int, int]] = []
            for s, e in windows:
                if not merged or s > merged[-1][1]:
                    merged.append((s, e))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))

            excerpt_lines: list[str] = []
            for s, e in merged[:20]:
                excerpt_lines.extend(lines[s:e])
                excerpt_lines.append("...")  # separator between windows

            excerpt = "\n".join(excerpt_lines).rstrip()
            blocks.append(f"### {fp.name} ({count} matches)\n\n```text\n{excerpt}\n```\n")

        return "\n".join(blocks).rstrip()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ------------------------------
# Log collection helper
# ------------------------------


async def _collect_log_lines(source: Any, max_lines: int, timeout_s: float) -> str:
    """Drain at most ``max_lines`` lines from an async log iterator within ``timeout_s``.

    Returns the joined lines with a trailing marker when the stream was cut short, so the caller
    can tell "that's all of it" apart from "there is more". Partial output survives the timeout:
    lines land in a list owned by this function, not by the cancelled inner coroutine.
    """
    lines: list[str] = []
    truncated = False

    async def _drain() -> None:
        nonlocal truncated
        async for line in source:
            lines.append(line)
            if len(lines) >= max_lines:
                truncated = True
                return

    timed_out = False
    try:
        await asyncio.wait_for(_drain(), timeout=timeout_s)
    except asyncio.TimeoutError:
        timed_out = True

    body = "\n".join(lines)
    if truncated:
        body += f"\n...truncated at {max_lines} lines"
    if timed_out:
        body += f"\n...stopped after {timeout_s:g}s; {len(lines)} lines collected so far"
    if not lines and not timed_out:
        return "No log lines were returned."
    return body


# ------------------------------
# Tool definitions
# ------------------------------


def build_tool_functions(env: FlyteMCPAppEnvironment) -> dict[str, Any]:
    """Build every MCP tool bound to ``env``, keyed by tool name.

    Returns *all* tools regardless of which ones are enabled; the caller decides what to
    register (see ``FlyteMCPAppEnvironment._create_mcp_server``). Building them all keeps the
    "declared in TOOL_REGISTRY but never implemented" check meaningful.
    """
    # ------------------------------
    # task
    # ------------------------------

    async def run_task(
        domain: str,
        project: str,
        name: str,
        inputs: dict,
        version: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Launch a registered task and return immediately with the new run's name and URL.

        Use this to start work; poll `get_run` or block with `wait_for_run` to see the
        outcome. `inputs` is a dict of the task's input names to values. Omit `version` to
        use the latest registered version.
        """
        if not _is_task_allowed(env.task_allowlist, domain, project, name):
            raise ToolError(f"Task {domain}/{project}/{name} is not allowlisted.")
        with env._scoped(project, domain):
            tk = flyte.remote.Task.get(
                project=project,
                domain=domain,
                name=name,
                version=version,
                auto_version="latest" if version is None else None,
            )  # type: ignore[arg-type]
            r = await flyte.run.aio(tk, **inputs)
            return {"url": r.url, "name": r.name}

    async def get_task(
        domain: str, project: str, name: str, version: str | None = None, ctx: MCPContext | None = None
    ) -> dict:
        """Describe a registered task: its interface, cache settings and secrets.

        Call this before `run_task` to learn which inputs are required and which have
        defaults. Omit `version` to describe the latest registered version.
        """
        if not _is_task_allowed(env.task_allowlist, domain, project, name):
            raise ToolError(f"Task {domain}/{project}/{name} is not allowlisted.")
        with env._scoped(project, domain):
            lazy = flyte.remote.Task.get(
                project=project,
                domain=domain,
                name=name,
                version=version,
                auto_version="latest" if version is None else None,
            )  # type: ignore[arg-type]
            td = await lazy.fetch.aio()
            iface = td.interface
            return {
                "name": td.name,
                "version": td.version,
                "task_type": td.task_type,
                # `signature` is the human-readable form; `inputs`/`outputs` are what you build
                # a `run_task(inputs=...)` payload from.
                "interface": {
                    "signature": str(iface),
                    "inputs": {k: _type_name(t) for k, (t, _) in iface.inputs.items()},
                    "outputs": {k: _type_name(t) for k, t in iface.outputs.items()},
                },
                "required_args": td.required_args,
                "default_input_args": td.default_input_args,
                "cache": {
                    "behavior": td.cache.behavior,
                    "version_override": td.cache.version_override,
                    "serialize": td.cache.serialize,
                },
                "secrets": td.secrets,
            }

    async def list_tasks(
        project: str | None = None,
        domain: str | None = None,
        limit: int = 100,
        entrypoint: bool | None = None,
        ctx: MCPContext | None = None,
    ) -> list[dict]:
        """List the tasks registered in a project/domain.

        Use this to discover what can be run. Set `entrypoint=True` to see only top-level
        tasks (the ones normally launched directly) rather than every sub-task. Only
        allowlisted tasks are listed, so this matches what `run_task` will accept.
        """
        allowlist = env.task_allowlist
        # The allowlist is applied here, not by the server, so ask for more than `limit` when one
        # is configured -- otherwise the server truncates first and allowed tasks disappear.
        scan_limit = limit if allowlist is None else max(limit, ALLOWLIST_SCAN_LIMIT)
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            tasks = cast(Any, flyte.remote.Task.listall).aio(
                project=resolved_project,
                domain=resolved_domain,
                sort_by=("created_at", "desc"),
                limit=scan_limit,
                entrypoint=entrypoint,
            )
            out: list[dict] = []
            async for t in tasks:
                name = getattr(t, "name", None)
                if not _is_task_allowed(allowlist, resolved_domain, resolved_project, name or ""):
                    continue
                if hasattr(t, "to_dict"):
                    out.append(t.to_dict())
                else:
                    out.append({"name": name, "version": getattr(t, "version", None)})
                if len(out) >= limit:
                    break
            return out

    # ------------------------------
    # run
    # ------------------------------

    async def get_run(
        name: str, project: str | None = None, domain: str | None = None, ctx: MCPContext | None = None
    ) -> dict:
        """Get a run's current phase plus its console URL.

        This is the cheap status check — it does not fetch inputs, outputs or logs. Use
        `get_run_io` for data and `get_logs` for output.
        """
        with env._scoped(project, domain):
            run = await flyte.remote.Run.get.aio(name=name)
            return {"name": run.name, "phase": _phase_str(run.phase), "url": run.url, "done": run.done()}

    async def wait_for_run(
        name: str,
        timeout_s: float | None = None,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Block until a run reaches a terminal phase, then report it.

        Prefer this over polling `get_run` in a loop. Always pass `timeout_s` for runs that
        might be long — without it this waits indefinitely. On timeout this does *not* error:
        it returns the run's current, non-terminal status with `timed_out: true`, so you can
        decide whether to wait again or `abort_run`.
        """
        with env._scoped(project, domain):
            run = await flyte.remote.Run.get.aio(name=name)
            timed_out = False
            try:
                await asyncio.wait_for(run.wait.aio(quiet=True), timeout=timeout_s)
            except asyncio.TimeoutError:
                timed_out = True
                # `wait` was cancelled mid-stream, so the local copy is stale; re-fetch the run
                # to report the phase it is actually in rather than the one we started from.
                run = await flyte.remote.Run.get.aio(name=name)

            result: dict[str, Any] = {
                "name": run.name,
                "phase": _phase_str(run.phase),
                "url": run.url,
                "done": run.done(),
            }
            if timed_out:
                result["timed_out"] = True
                result["message"] = (
                    f"Run '{name}' was still {_phase_str(run.phase)} after {timeout_s:g}s and is "
                    f"still going. Call wait_for_run again to keep waiting, or abort_run to stop it."
                )
            return result

    async def get_run_io(
        name: str, project: str | None = None, domain: str | None = None, ctx: MCPContext | None = None
    ) -> dict:
        """Get a run's inputs, and its outputs once it has finished.

        `outputs` is null while the run is still going. If the outputs exist but cannot be
        decoded (an unreconstructable output type, for example), `outputs_error` explains
        why instead of the outputs silently coming back null.
        """
        with env._scoped(project, domain):
            run = await flyte.remote.Run.get.aio(name=name)
            inputs = await run.inputs.aio()
            outputs = None
            outputs_error: str | None = None
            if run.done():
                try:
                    outputs = await run.outputs.aio()
                except Exception as e:
                    outputs_error = f"{type(e).__name__}: {e}"
            inputs_dict = (
                dict(inputs)
                if inputs is not None and hasattr(inputs, "keys")
                else (inputs.to_dict() if hasattr(inputs, "to_dict") else inputs)
            )
            outputs_dict: Any = None
            if outputs is not None:
                outputs_dict = (
                    outputs.named_outputs
                    if hasattr(outputs, "named_outputs")
                    else (outputs.to_dict() if hasattr(outputs, "to_dict") else list(outputs))
                )
            result = {"name": run.name, "inputs": inputs_dict, "outputs": outputs_dict}
            if outputs_error is not None:
                result["outputs_error"] = outputs_error
            return result

    async def abort_run(
        name: str,
        reason: str | None = None,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Terminate a run and every action under it. Work already done is not rolled back.

        Aborting an already-finished run is a no-op, so this is safe to retry.
        """
        with env._scoped(project, domain):
            run = await flyte.remote.Run.get.aio(name=name)
            if reason:
                await run.abort.aio(reason=reason)
            else:
                await run.abort.aio()
            return {"name": run.name, "aborted": True}

    async def list_runs(
        task_name: str | None = None,
        limit: int = 100,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> list[dict]:
        """List recent runs in a project/domain, newest first.

        Pass `task_name` to see only the runs of one task — useful for "did my last run of
        X succeed?" questions.
        """
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            runs = cast(Any, flyte.remote.Run.listall).aio(
                task_name=task_name,
                sort_by=("created_at", "desc"),
                limit=limit,
                project=resolved_project,
                domain=resolved_domain,
            )
            out: list[dict] = []
            async for r in runs:
                if hasattr(r, "to_dict"):
                    out.append(r.to_dict())
                else:
                    out.append({"name": getattr(r, "name", None), "url": getattr(r, "url", None)})
                if len(out) >= limit:
                    break
            return out

    async def rerun_run(
        run_name: str, project: str | None = None, domain: str | None = None, ctx: MCPContext | None = None
    ) -> dict:
        """Re-run a previous run's root action with exactly the same inputs and code.

        Use this to retry a failed run without knowing its inputs or having the source
        locally. Returns the *new* run's name and URL; the original is untouched. The task
        behind the run must be allowlisted, exactly as it would be for `run_task`.
        """
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            # A rerun launches the prior run's task, so it has to clear the same gate as run_task.
            # Only pay for the lookup when there is an allowlist to check against.
            if env.task_allowlist is not None:
                prior = await flyte.remote.Run.get.aio(name=run_name)
                task_name = getattr(prior.action, "task_name", None)
                if task_name is None:
                    raise ToolError(
                        f"Cannot determine which task run '{run_name}' launched, so it cannot be "
                        f"checked against the task allowlist."
                    )
                if not _is_task_allowed(env.task_allowlist, resolved_domain, resolved_project, task_name):
                    raise ToolError(f"Task {resolved_domain}/{resolved_project}/{task_name} is not allowlisted.")
            r = await flyte.rerun.aio(run_name)
            return {"url": r.url, "name": r.name}

    # ------------------------------
    # action
    # ------------------------------

    async def list_actions(
        run_name: str,
        limit: int = 100,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> list[dict]:
        """List the actions (individual task executions) inside a run.

        This is how you find which step of a multi-step run failed: scan the phases, then
        call `get_action` or `get_logs` on the interesting one.
        """
        with env._scoped(project, domain):
            out: list[dict] = []
            async for a in cast(Any, flyte.remote.Action.listall).aio(for_run_name=run_name):
                started_at = _ts(a.pb2.status, "start_time")
                ended_at = _ts(a.pb2.status, "end_time")
                out.append(
                    {
                        "name": a.name,
                        "task_name": a.task_name,
                        "phase": _phase_str(a.phase),
                        "started_at": started_at,
                        "ended_at": ended_at,
                    }
                )
                if len(out) >= limit:
                    break
            return out

    async def get_action(
        run_name: str,
        action_name: str,
        attempt: int | None = None,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Get one action's phase, attempts, failure details and per-phase timing.

        Use this to answer "why did this step fail?" (`error_info`/`abort_info`) and "where
        did the time go?" (`timing`: seconds queued, initializing, running, total).
        `logs_available` tells you whether `get_logs` will return anything yet. Pass
        `attempt` (1-indexed) to describe a specific retry instead of the latest one — both
        the timing and `logs_available` then refer to that attempt.
        """
        with env._scoped(project, domain):
            details = await flyte.remote.ActionDetails.get.aio(run_name=run_name, name=action_name)
            from flyte.models import ActionPhase

            if attempt is None:
                durations = details.phase_durations
                total_s = _seconds(details.runtime)
            else:
                # `phase_durations` is the latest attempt only, so an explicit attempt has to go
                # through the transitions directly or the timing would describe a different retry.
                transitions = details.get_phase_transitions(attempt)
                durations = {t.phase: t.duration for t in transitions}
                total_s = _seconds(sum((t.duration for t in transitions), timedelta())) if transitions else None

            url = None
            try:
                from flyte._initialize import get_client

                action_id = details.action_id
                url = get_client().console.run_url(
                    project=action_id.run.project,
                    domain=action_id.run.domain,
                    run_name=action_id.run.name,
                )
            except Exception:  # pragma: no cover - console URL is best-effort metadata
                url = None

            return {
                "name": details.name,
                "run_name": details.run_name,
                "task_name": details.task_name,
                "phase": _phase_str(details.phase),
                "attempts": details.attempts,
                "error_info": _pb_to_dict(details.error_info),
                "abort_info": _pb_to_dict(details.abort_info),
                "timing": {
                    "attempt": attempt if attempt is not None else details.attempts,
                    "queued_s": _seconds(durations.get(ActionPhase.QUEUED)),
                    "initializing_s": _seconds(durations.get(ActionPhase.INITIALIZING)),
                    "running_s": _seconds(durations.get(ActionPhase.RUNNING)),
                    "total_s": total_s,
                },
                "logs_available": details.logs_available(attempt),
                "url": url,
            }

    async def abort_action(
        run_name: str,
        action_name: str,
        reason: str | None = None,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Terminate a single action within a run, leaving its siblings running.

        Use `abort_run` to stop the whole run instead. Aborting a finished action is a
        no-op, so this is safe to retry.
        """
        with env._scoped(project, domain):
            action = await flyte.remote.Action.get.aio(run_name=run_name, name=action_name)
            if reason:
                await action.abort.aio(reason=reason)
            else:
                await action.abort.aio()
            return {"run_name": run_name, "name": action_name, "aborted": True}

    # ------------------------------
    # logs
    # ------------------------------

    async def get_logs(
        run_name: str,
        action_name: str | None = None,
        attempt: int | None = None,
        max_lines: int = 200,
        filter_system: bool = True,
        wait_for_logs: bool = False,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> str:
        """Read an action's logs as text. Omit `action_name` for the run's root action.

        Returns at most `max_lines` lines (hard cap 2000) and gives up after ~30s, marking
        the output when it was cut short. If the logs are not up yet this returns a short
        status string right away rather than blocking; pass `wait_for_logs=True` only when
        you actually want to wait for a queued action to start.
        """
        bounded_lines = max(1, min(int(max_lines), MAX_LOG_LINES))
        with env._scoped(project, domain):
            if action_name is None:
                target: Any = await flyte.remote.Run.get.aio(name=run_name)
                details = await target.action.details()
                label = f"run '{run_name}'"
            else:
                target = await flyte.remote.Action.get.aio(run_name=run_name, name=action_name)
                details = await target.details()
                label = f"action '{action_name}' in run '{run_name}'"

            if not wait_for_logs and not details.logs_available(attempt):
                return (
                    f"Logs are not available yet for {label} (phase {_phase_str(details.phase)}). "
                    f"Call get_logs again once it is running, or pass wait_for_logs=true to block."
                )

            source = cast(Any, target.get_logs).aio(attempt=attempt, filter_system=filter_system)
            return await _collect_log_lines(source, bounded_lines, LOG_COLLECT_TIMEOUT_S)

    # ------------------------------
    # app
    # ------------------------------

    async def get_app(
        name: str, project: str | None = None, domain: str | None = None, ctx: MCPContext | None = None
    ) -> dict:
        """Get a deployed app's status, public endpoint and console URL."""
        if not _is_app_allowed(env.app_allowlist, name):
            raise ToolError(f"App {name} is not allowlisted.")
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            app = await flyte.remote.App.get.aio(name=name, project=resolved_project, domain=resolved_domain)
            return (
                app.to_dict()
                if hasattr(app, "to_dict")
                else {"name": getattr(app, "name", None), "endpoint": getattr(app, "endpoint", None)}
            )

    async def list_apps(
        limit: int = 100,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> list[dict]:
        """List deployed apps with their deployment status and public endpoints.

        Use this to find an app's name before calling `get_app`, `activate_app` or
        `deactivate_app`.
        """
        from flyteidl2.app import app_definition_pb2

        allowlist = env.app_allowlist
        # See list_tasks: the server truncates before the allowlist is applied, so scan wider.
        scan_limit = limit if allowlist is None else max(limit, ALLOWLIST_SCAN_LIMIT)
        with env._scoped(project, domain):
            out: list[dict] = []
            async for app in cast(Any, flyte.remote.App.listall).aio(sort_by=("created_at", "desc"), limit=scan_limit):
                if not _is_app_allowed(allowlist, app.name):
                    continue
                status = app_definition_pb2.Status.DeploymentStatus.Name(app.deployment_status)
                url = None
                try:
                    url = app.url
                except Exception:  # pragma: no cover - console URL is best-effort metadata
                    url = None
                out.append(
                    {
                        "name": app.name,
                        "status": status.removeprefix("DEPLOYMENT_STATUS_"),
                        "endpoint": app.endpoint,
                        "url": url,
                    }
                )
                if len(out) >= limit:
                    break
            return out

    async def activate_app(
        name: str, project: str | None = None, domain: str | None = None, ctx: MCPContext | None = None
    ) -> dict:
        """Start a deployed app and wait for it to become active. Already-active apps are unchanged."""
        if not _is_app_allowed(env.app_allowlist, name):
            raise ToolError(f"App {name} is not allowlisted.")
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            app = await flyte.remote.App.get.aio(name=name, project=resolved_project, domain=resolved_domain)
            activated = await app.activate.aio(wait=True)
            return (
                activated.to_dict()
                if hasattr(activated, "to_dict")
                else {"name": getattr(activated, "name", None), "activated": True}
            )

    async def deactivate_app(
        name: str, project: str | None = None, domain: str | None = None, ctx: MCPContext | None = None
    ) -> dict:
        """Stop a deployed app and wait for it to shut down. Its public endpoint stops serving."""
        if not _is_app_allowed(env.app_allowlist, name):
            raise ToolError(f"App {name} is not allowlisted.")
        with env._scoped(project, domain) as (resolved_project, resolved_domain):
            app = await flyte.remote.App.get.aio(name=name, project=resolved_project, domain=resolved_domain)
            deactivated = await app.deactivate.aio(wait=True)
            return (
                deactivated.to_dict()
                if hasattr(deactivated, "to_dict")
                else {"name": getattr(deactivated, "name", None), "deactivated": True}
            )

    # ------------------------------
    # trigger
    # ------------------------------

    async def list_triggers(
        task_name: str | None = None,
        limit: int = 100,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> list[dict]:
        """List triggers (schedules) with their task, automation spec and active state.

        Pass `task_name` to narrow to one task's triggers. Use this to answer "what is
        scheduled?" and to find the exact trigger name for activate/deactivate.
        """
        from flyte.remote._trigger import _describe_automation

        allowlist = env.trigger_allowlist
        # See list_tasks: the server truncates before the allowlist is applied, so scan wider.
        scan_limit = limit if allowlist is None else max(limit, ALLOWLIST_SCAN_LIMIT)
        with env._scoped(project, domain):
            out: list[dict] = []
            async for t in cast(Any, flyte.remote.Trigger.listall).aio(task_name=task_name, limit=scan_limit):
                if not _is_trigger_allowed(allowlist, t.task_name, t.name):
                    continue
                url = None
                try:
                    url = t.url
                except Exception:  # pragma: no cover - console URL is best-effort metadata
                    url = None
                out.append(
                    {
                        "name": t.name,
                        "task_name": t.task_name,
                        "is_active": t.is_active,
                        "automation": _describe_automation(t.automation_spec),
                        "url": url,
                    }
                )
                if len(out) >= limit:
                    break
            return out

    async def get_trigger(
        trigger_name: str,
        task_name: str,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Get one trigger: whether it is active, what fires it, and its console URL."""
        if not _is_trigger_allowed(env.trigger_allowlist, task_name, trigger_name):
            raise ToolError(f"Trigger {task_name}/{trigger_name} is not allowlisted.")
        from flyte.remote._trigger import _describe_automation

        with env._scoped(project, domain):
            details = await flyte.remote.Trigger.get.aio(name=trigger_name, task_name=task_name)
            url = None
            try:
                from flyte._initialize import get_client

                trigger_name_pb = details.id.name
                url = get_client().console.trigger_url(
                    project=trigger_name_pb.project,
                    domain=trigger_name_pb.domain,
                    task_name=trigger_name_pb.task_name,
                    trigger_name=trigger_name_pb.name,
                )
            except Exception:  # pragma: no cover - console URL is best-effort metadata
                url = None
            return {
                "name": details.name,
                "task_name": details.task_name,
                "is_active": details.is_active,
                "automation": _describe_automation(details.automation_spec),
                "url": url,
            }

    async def activate_trigger(
        task_name: str,
        trigger_name: str,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Activate a trigger so its schedule starts firing runs again.

        Activating an already-active trigger is a no-op, so this is safe to retry.
        """
        if not _is_trigger_allowed(env.trigger_allowlist, task_name, trigger_name):
            raise ToolError(f"Trigger {task_name}/{trigger_name} is not allowlisted.")
        with env._scoped(project, domain):
            await cast(Any, flyte.remote.Trigger.update).aio(name=trigger_name, task_name=task_name, active=True)
            return {"task_name": task_name, "name": trigger_name, "activated": True}

    async def deactivate_trigger(
        task_name: str,
        trigger_name: str,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Pause a trigger so its schedule stops firing new runs. Runs in flight are unaffected."""
        if not _is_trigger_allowed(env.trigger_allowlist, task_name, trigger_name):
            raise ToolError(f"Trigger {task_name}/{trigger_name} is not allowlisted.")
        with env._scoped(project, domain):
            await cast(Any, flyte.remote.Trigger.update).aio(name=trigger_name, task_name=task_name, active=False)
            return {"task_name": task_name, "name": trigger_name, "deactivated": True}

    # ------------------------------
    # project
    # ------------------------------

    async def list_projects(limit: int = 100, ctx: MCPContext | None = None) -> list[dict]:
        """List the active projects on this control plane.

        Start here when you do not know which project to work in; the `id` values are what
        other tools take as their `project` argument.
        """
        from flyteidl2.project import project_service_pb2

        out: list[dict] = []
        async for p in cast(Any, flyte.remote.Project.listall).aio():
            out.append(
                {
                    "id": p.pb2.id,
                    "name": p.pb2.name,
                    "description": p.pb2.description,
                    "state": project_service_pb2.ProjectState.Name(p.pb2.state),
                }
            )
            if len(out) >= limit:
                break
        return out

    async def get_project(name: str, ctx: MCPContext | None = None) -> dict:
        """Get one project's metadata, including its configured domains."""
        from flyteidl2.project import project_service_pb2

        p = await flyte.remote.Project.get.aio(name=name)
        return {
            "id": p.pb2.id,
            "name": p.pb2.name,
            "description": p.pb2.description,
            "state": project_service_pb2.ProjectState.Name(p.pb2.state),
            "domains": [d.id for d in p.pb2.domains],
            "labels": dict(p.pb2.labels.values) if p.pb2.HasField("labels") else {},
        }

    # ------------------------------
    # secret
    # ------------------------------

    async def list_secrets(
        limit: int = 100,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> list[dict]:
        """List the names and types of secrets in a project/domain.

        Secret *values* are never returned by any tool. Use this to check whether a secret a
        task needs already exists.
        """
        with env._scoped(project, domain):
            out: list[dict] = []
            async for s in cast(Any, flyte.remote.Secret.listall).aio(limit=min(100, max(1, limit))):
                out.append({"name": s.name, "type": s.type})
                if len(out) >= limit:
                    break
            return out

    async def create_secret(
        name: str,
        value: str,
        type: str = "regular",
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Create a secret in a project/domain. The value is written, never echoed back.

        `type` must be "regular"; "image_pull" secrets are scoped to a cluster pool rather
        than to a project/domain and cannot be created here. Creating a secret that already
        exists fails — delete it first if you mean to replace it.
        """
        # Secret.create treats *anything* that is not "regular" as an image_pull secret, which
        # then fails deeper down on the project/domain scope this server always sets. Reject the
        # typo here so the caller gets a sentence they can act on.
        if type != "regular":
            raise ToolError(
                f"Unsupported secret type '{type}'. Only 'regular' secrets can be created through this "
                f"server; image_pull secrets are scoped to a cluster pool, not to a project/domain."
            )
        with env._scoped(project, domain):
            await cast(Any, flyte.remote.Secret.create).aio(name=name, value=value, type=type)
            return {"name": name, "created": True}

    async def delete_secret(
        name: str,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Delete a secret from a project/domain. Tasks still referencing it will fail to start."""
        with env._scoped(project, domain):
            await cast(Any, flyte.remote.Secret.delete).aio(name)
            return {"name": name, "deleted": True}

    # ------------------------------
    # condition
    # ------------------------------

    async def list_conditions(
        run_name: str,
        limit: int = 50,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> list[dict]:
        """List the conditions of a run — the points where it is paused for external input.

        `expected_type` is the payload type `signal_condition` will coerce to. A run stuck in
        a paused phase usually has an unsignalled condition here.
        """
        from flyte.remote._condition import resolve_condition_expected_type

        with env._scoped(project, domain):
            out: list[dict] = []
            async for c in cast(Any, flyte.remote.Condition.listall).aio(run_name=run_name, limit=limit):
                expected = resolve_condition_expected_type(c.pb2)
                out.append(
                    {
                        "name": c.name,
                        "action_name": c.action_name,
                        "run_name": c.run_name,
                        "phase": c.phase,
                        "expected_type": expected.__name__ if expected is not None else None,
                    }
                )
                if len(out) >= limit:
                    break
            return out

    async def signal_condition(
        name: str,
        run_name: str,
        value: bool | int | float | str,
        project: str | None = None,
        domain: str | None = None,
        ctx: MCPContext | None = None,
    ) -> dict:
        """Resolve a waiting condition with a value, unblocking the run.

        `value` is coerced to the condition's declared type (see `list_conditions`), so the
        string "true" is accepted for a bool condition. This is a one-shot decision: signal
        the wrong value and the run proceeds down the wrong branch.
        """
        from flyte.remote._condition import coerce_condition_payload, resolve_condition_expected_type

        with env._scoped(project, domain):
            condition = await cast(Any, flyte.remote.Condition.get).aio(name, run_name=run_name)
            if condition is None:
                raise ToolError(f"Condition '{name}' was not found in run '{run_name}'.")
            expected = resolve_condition_expected_type(condition.pb2)
            payload = coerce_condition_payload(value, expected) if expected is not None else value
            await condition.signal.aio(payload)
            return {
                "name": condition.name,
                "run_name": run_name,
                "signaled": True,
                "value": payload,
            }

    # ------------------------------
    # identity
    # ------------------------------

    async def whoami(ctx: MCPContext | None = None) -> dict:
        """Report which identity, org and control plane this server is acting as.

        Use this first when a call fails with a permission or "not found" error — it tells
        you whether you are pointed at the tenant you think you are.
        """
        from flyte._initialize import _get_init_config

        user = await flyte.remote.User.get.aio()
        cfg = _get_init_config()
        endpoint = None
        if cfg is not None and cfg.client is not None:
            endpoint = getattr(cfg.client, "endpoint", None)
        return {
            "subject": user.subject(),
            "name": user.name(),
            "org": cfg.org if cfg is not None else None,
            "endpoint": endpoint,
            "project": cfg.project if cfg is not None else None,
            "domain": cfg.domain if cfg is not None else None,
        }

    # ------------------------------
    # search
    # ------------------------------

    async def search_flyte_sdk_examples(
        pattern: str, ctx: MCPContext | None = None, before_context_lines: int = 5, after_context_lines: int = 5
    ) -> str:
        """Search the bundled Flyte SDK examples for a regex pattern.

        Use this to find working code for an SDK API before writing a task. Returns the
        match count per file plus context excerpts from the top 3 files.
        """
        if env.sdk_examples_path is None:
            raise ToolError("sdk_examples_path is not configured for this MCP environment.")
        return await _search_files(
            pattern,
            env.sdk_examples_path,
            top_n=3,
            before_context_lines=before_context_lines,
            after_context_lines=after_context_lines,
        )

    async def search_flyte_docs_examples(
        pattern: str, ctx: MCPContext | None = None, before_context_lines: int = 5, after_context_lines: int = 5
    ) -> str:
        """Search the bundled Union docs examples for a regex pattern.

        Complements `search_flyte_sdk_examples` with the longer-form, documented examples.
        """
        if env.docs_examples_path is None:
            raise ToolError("docs_examples_path is not configured for this MCP environment.")
        return await _search_files(
            pattern,
            env.docs_examples_path,
            top_n=3,
            before_context_lines=before_context_lines,
            after_context_lines=after_context_lines,
        )

    async def search_full_docs(
        pattern: str, ctx: MCPContext | None = None, before_context_lines: int = 20, after_context_lines: int = 20
    ) -> str:
        """Search the full Flyte/Union documentation text for a regex pattern.

        Use this for conceptual questions ("how does caching work?") where the answer is
        prose rather than code.
        """
        if env.full_docs_path is None:
            raise ToolError("full_docs_path is not configured for this MCP environment.")
        return await _search_files(
            pattern,
            env.full_docs_path,
            top_n=3,
            before_context_lines=before_context_lines,
            after_context_lines=after_context_lines,
        )

    # The registration table: tool name -> handler. Paired with ``TOOL_REGISTRY`` (group +
    # annotations) in ``_flyte_mcp_app.py``, which is what decides what actually gets registered.
    tool_functions: dict[str, Any] = {
        "run_task": run_task,
        "get_task": get_task,
        "list_tasks": list_tasks,
        "get_run": get_run,
        "get_run_io": get_run_io,
        "abort_run": abort_run,
        "list_runs": list_runs,
        "wait_for_run": wait_for_run,
        "rerun_run": rerun_run,
        "list_actions": list_actions,
        "get_action": get_action,
        "abort_action": abort_action,
        "get_logs": get_logs,
        "get_app": get_app,
        "list_apps": list_apps,
        "activate_app": activate_app,
        "deactivate_app": deactivate_app,
        "list_triggers": list_triggers,
        "get_trigger": get_trigger,
        "activate_trigger": activate_trigger,
        "deactivate_trigger": deactivate_trigger,
        "list_projects": list_projects,
        "get_project": get_project,
        "list_secrets": list_secrets,
        "create_secret": create_secret,
        "delete_secret": delete_secret,
        "list_conditions": list_conditions,
        "signal_condition": signal_condition,
        "whoami": whoami,
        "search_flyte_sdk_examples": search_flyte_sdk_examples,
        "search_flyte_docs_examples": search_flyte_docs_examples,
        "search_full_docs": search_full_docs,
    }

    return tool_functions
