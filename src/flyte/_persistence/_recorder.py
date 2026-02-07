from __future__ import annotations

from typing import Any


class RunRecorder:
    """Unified proxy that delegates recording events to the TUI tracker and/or
    the SQLite persistence layer (RunStore).

    The controller only talks to this single object — no more interleaved
    ``if tracker`` / ``if persist`` conditionals.

    Lazy-imports are used so that neither ``RunStore`` nor
    ``literal_string_repr`` are imported at module level.
    """

    def __init__(
        self,
        tracker: Any | None = None,
        persist: bool = False,
        run_name: str | None = None,
    ) -> None:
        self._tracker = tracker
        self._persist = persist and run_name is not None
        self._run_name: str = run_name or ""

    @property
    def is_active(self) -> bool:
        """True if at least one recording backend is enabled."""
        return self._tracker is not None or self._persist

    def get_action(self, action_id: str) -> Any:
        """Delegate to the tracker, or return None when tracker is absent."""
        if self._tracker is not None:
            return self._tracker.get_action(action_id)
        return None

    # ------------------------------------------------------------------
    # Sub-action lifecycle (called by LocalController)
    # ------------------------------------------------------------------

    def record_start(
        self,
        *,
        action_id: str,
        task_name: str,
        parent_id: str | None = None,
        short_name: str | None = None,
        inputs: dict | None = None,
        output_path: str | None = None,
        has_report: bool = False,
        cache_enabled: bool = False,
        cache_hit: bool = False,
        context: dict | None = None,
        group: str | None = None,
        log_links: list[tuple[str, str]] | None = None,
    ) -> None:
        if self._tracker is not None:
            self._tracker.record_start(
                action_id=action_id,
                task_name=task_name,
                short_name=short_name,
                parent_id=parent_id,
                inputs=inputs,
                output_path=output_path,
                has_report=has_report,
                cache_enabled=cache_enabled,
                cache_hit=cache_hit,
                context=context,
                group=group,
                log_links=log_links,
            )

        if self._persist:
            from flyte._persistence._run_store import RunStore

            persist_parent = "a0" if parent_id is None else parent_id
            RunStore.record_start_sync(
                run_name=self._run_name,
                action_name=action_id,
                task_name=task_name,
                parent_id=persist_parent,
                short_name=short_name,
                inputs=inputs,
                output_path=output_path,
                has_report=bool(has_report),
                cache_enabled=cache_enabled,
                cache_hit=cache_hit,
                context=context,
                group_name=group,
                log_links=log_links,
            )

    def record_complete(self, *, action_id: str, outputs: Any = None) -> None:
        # Convert outputs to a display representation once, so both backends
        # receive the same pre-formatted data.
        display: Any = None
        if outputs is not None:
            display = self._to_display(outputs)

        if self._tracker is not None:
            self._tracker.record_complete(action_id=action_id, outputs=display)

        if self._persist:
            from flyte._persistence._run_store import RunStore

            RunStore.record_complete_sync(
                run_name=self._run_name,
                action_name=action_id,
                outputs=repr(display) if display is not None else None,
            )

    @staticmethod
    def _to_display(outputs: Any) -> Any:
        """Convert raw outputs to a display-friendly representation.

        Handles ``Outputs`` proto wrappers, plain strings, and arbitrary
        literal trees.  Falls back to ``repr()`` when ``literal_string_repr``
        is unavailable or raises.
        """
        try:
            from flyte._internal.runtime.io import Outputs
            from flyte.types._string_literals import literal_string_repr

            if isinstance(outputs, Outputs):
                return literal_string_repr(outputs.proto_outputs)
            if isinstance(outputs, str):
                return outputs
            return literal_string_repr(outputs)
        except Exception:
            return repr(outputs)

    def record_failure(self, *, action_id: str, error: str) -> None:
        if self._tracker is not None:
            self._tracker.record_failure(action_id=action_id, error=error)

        if self._persist:
            from flyte._persistence._run_store import RunStore

            RunStore.record_failure_sync(
                run_name=self._run_name,
                action_name=action_id,
                error=error,
            )

    # ------------------------------------------------------------------
    # Root "a0" action (called by _run.py — persistence only)
    # ------------------------------------------------------------------

    def record_root_start(self, *, task_name: str) -> None:
        if self._persist:
            from flyte._persistence._run_store import RunStore

            RunStore.record_start_sync(
                run_name=self._run_name,
                action_name="a0",
                task_name=task_name,
                parent_id=None,
            )

    def record_root_complete(self) -> None:
        if self._persist:
            from flyte._persistence._run_store import RunStore

            RunStore.record_complete_sync(
                run_name=self._run_name,
                action_name="a0",
            )

    def record_root_failure(self, *, error: str) -> None:
        if self._persist:
            from flyte._persistence._run_store import RunStore

            RunStore.record_failure_sync(
                run_name=self._run_name,
                action_name="a0",
                error=error,
            )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def initialize_persistence() -> None:
        from flyte._persistence._run_store import RunStore

        RunStore.initialize_sync()
