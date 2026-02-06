import json
import time
from dataclasses import dataclass

from flyte._persistence._db import LocalDB


@dataclass
class RunRecord:
    run_name: str
    action_name: str
    task_name: str | None = None
    status: str = "running"
    inputs: str | None = None
    outputs: str | None = None
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    parent_id: str | None = None
    short_name: str | None = None
    output_path: str | None = None
    cache_enabled: bool = False
    cache_hit: bool = False
    has_report: bool = False
    context: str | None = None
    group_name: str | None = None
    log_links: str | None = None


def _json_or_none(obj) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, default=repr)


class RunStore:
    """Persistence layer for local run metadata.

    All public methods are sync (called from controller threads or TUI).
    Write operations are serialized via LocalDB._write_lock.
    """

    @staticmethod
    def initialize_sync():
        LocalDB.initialize_sync()

    # -- Write methods (sync) --

    @staticmethod
    def record_start_sync(
        run_name: str,
        action_name: str,
        task_name: str | None = None,
        parent_id: str | None = None,
        short_name: str | None = None,
        inputs: dict | None = None,
        output_path: str | None = None,
        cache_enabled: bool = False,
        cache_hit: bool = False,
        has_report: bool = False,
        context: dict | None = None,
        group_name: str | None = None,
        log_links: list[tuple[str, str]] | None = None,
    ) -> None:
        inputs_json = _json_or_none(inputs)
        context_json = _json_or_none(context)
        log_links_json = _json_or_none(log_links)
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_name, action_name, task_name, status, inputs, start_time,
                    parent_id, short_name, output_path, cache_enabled, cache_hit,
                    has_report, context, group_name, log_links)
                   VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_name,
                    action_name,
                    task_name,
                    inputs_json,
                    time.time(),
                    parent_id,
                    short_name,
                    output_path,
                    int(cache_enabled),
                    int(cache_hit),
                    int(has_report),
                    context_json,
                    group_name,
                    log_links_json,
                ),
            )
            conn.commit()

    @staticmethod
    def record_complete_sync(
        run_name: str,
        action_name: str,
        outputs: str | None = None,
    ) -> None:
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            conn.execute(
                "UPDATE runs SET status='succeeded', outputs=?, end_time=? WHERE run_name=? AND action_name=?",
                (outputs, time.time(), run_name, action_name),
            )
            conn.commit()

    @staticmethod
    def record_failure_sync(
        run_name: str,
        action_name: str,
        error: str | None = None,
    ) -> None:
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            conn.execute(
                "UPDATE runs SET status='failed', error=?, end_time=? WHERE run_name=? AND action_name=?",
                (error, time.time(), run_name, action_name),
            )
            conn.commit()

    # -- Read methods (sync) --

    @staticmethod
    def list_runs_sync() -> list[RunRecord]:
        """List top-level runs (action_name='a0') ordered by start_time DESC."""
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            cursor = conn.execute(
                "SELECT * FROM runs WHERE action_name='a0' ORDER BY start_time DESC"
            )
            return [RunStore._row_to_record(row) for row in cursor.fetchall()]

    @staticmethod
    def list_actions_for_run_sync(run_name: str) -> list[RunRecord]:
        """List all actions for a given run."""
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            cursor = conn.execute(
                "SELECT * FROM runs WHERE run_name=? ORDER BY start_time ASC",
                (run_name,),
            )
            return [RunStore._row_to_record(row) for row in cursor.fetchall()]

    @staticmethod
    def get_action_sync(run_name: str, action_name: str) -> RunRecord | None:
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            cursor = conn.execute(
                "SELECT * FROM runs WHERE run_name=? AND action_name=?",
                (run_name, action_name),
            )
            row = cursor.fetchone()
            return RunStore._row_to_record(row) if row else None

    # -- Management --

    @staticmethod
    def delete_run_sync(run_name: str) -> None:
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            conn.execute("DELETE FROM runs WHERE run_name=?", (run_name,))
            conn.commit()

    @staticmethod
    def delete_runs_sync(run_names: list[str]) -> None:
        if not run_names:
            return
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            placeholders = ",".join("?" for _ in run_names)
            conn.execute(f"DELETE FROM runs WHERE run_name IN ({placeholders})", run_names)
            conn.commit()

    @staticmethod
    def clear_sync() -> None:
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            conn.execute("DELETE FROM runs")
            conn.commit()

    # -- Internal --

    @staticmethod
    def _row_to_record(row: tuple) -> RunRecord:
        return RunRecord(
            run_name=row[0],
            action_name=row[1],
            task_name=row[2],
            status=row[3],
            inputs=row[4],
            outputs=row[5],
            error=row[6],
            start_time=row[7],
            end_time=row[8],
            parent_id=row[9],
            short_name=row[10],
            output_path=row[11],
            cache_enabled=bool(row[12]),
            cache_hit=bool(row[13]),
            has_report=bool(row[14]),
            context=row[15],
            group_name=row[16],
            log_links=row[17],
        )
