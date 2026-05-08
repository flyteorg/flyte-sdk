import json
import time
from dataclasses import dataclass
from typing import ClassVar

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
    disable_run_cache: bool = False
    has_report: bool = False
    context: str | None = None
    group_name: str | None = None
    log_links: str | None = None
    attempt_count: int = 0
    attempts_json: str | None = None
    max_attempts_used: int = 1
    retried_actions: int = 0


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
        disable_run_cache: bool = False,
        has_report: bool = False,
        context: dict | None = None,
        group_name: str | None = None,
        log_links: list[tuple[str, str]] | None = None,
        attempt_count: int = 0,
        attempts_json: str | None = None,
    ) -> None:
        inputs_json = _json_or_none(inputs)
        context_json = _json_or_none(context)
        log_links_json = _json_or_none(log_links)
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_name, action_name, task_name, status, inputs, start_time,
                    parent_id, short_name, output_path, cache_enabled, cache_hit, disable_run_cache,
                    has_report, context, group_name, log_links, attempt_count, attempts_json)
                   VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    int(disable_run_cache),
                    int(has_report),
                    context_json,
                    group_name,
                    log_links_json,
                    attempt_count,
                    attempts_json,
                ),
            )
            conn.commit()

    @staticmethod
    def _load_attempts_json(raw: str | None) -> list[dict]:
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
        return []

    @staticmethod
    def _update_attempt_sync(
        run_name: str,
        action_name: str,
        attempt_num: int,
        *,
        status: str,
        outputs: str | None = None,
        error: str | None = None,
        end_time: float | None = None,
    ) -> None:
        with LocalDB._write_lock:
            conn = LocalDB.get_sync()
            row = conn.execute(
                "SELECT attempts_json, attempt_count FROM runs WHERE run_name=? AND action_name=?",
                (run_name, action_name),
            ).fetchone()
            attempts = RunStore._load_attempts_json(row[0] if row else None)
            existing_count = int(row[1] or 0) if row else 0
            now = time.time()

            record = next((a for a in attempts if int(a.get("attempt_num", -1)) == attempt_num), None)
            if record is None:
                record = {
                    "attempt_num": attempt_num,
                    "start_time": now,
                }
                attempts.append(record)
            record["status"] = status
            record["attempt_num"] = attempt_num
            record["outputs"] = outputs
            record["error"] = error
            if status == "running":
                record["start_time"] = record.get("start_time", now)
                record["end_time"] = None
            else:
                record["end_time"] = end_time if end_time is not None else now

            attempts.sort(key=lambda a: int(a.get("attempt_num", 0)))
            attempt_count = max(existing_count, attempt_num, len(attempts))
            conn.execute(
                "UPDATE runs SET attempt_count=?, attempts_json=? WHERE run_name=? AND action_name=?",
                (attempt_count, json.dumps(attempts, default=repr), run_name, action_name),
            )
            conn.commit()

    @staticmethod
    def record_attempt_start_sync(
        run_name: str,
        action_name: str,
        attempt_num: int,
    ) -> None:
        RunStore._update_attempt_sync(
            run_name=run_name,
            action_name=action_name,
            attempt_num=attempt_num,
            status="running",
        )

    @staticmethod
    def record_attempt_complete_sync(
        run_name: str,
        action_name: str,
        attempt_num: int,
        outputs: str | None = None,
    ) -> None:
        RunStore._update_attempt_sync(
            run_name=run_name,
            action_name=action_name,
            attempt_num=attempt_num,
            status="succeeded",
            outputs=outputs,
        )

    @staticmethod
    def record_attempt_failure_sync(
        run_name: str,
        action_name: str,
        attempt_num: int,
        error: str | None = None,
    ) -> None:
        RunStore._update_attempt_sync(
            run_name=run_name,
            action_name=action_name,
            attempt_num=attempt_num,
            status="failed",
            error=error,
        )

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

    # Columns that are safe to use in ORDER BY.
    _SORTABLE_COLUMNS: ClassVar[set[str]] = {
        "start_time",
        "run_name",
        "task_name",
        "status",
        "duration",
    }

    _VALID_STATUSES: ClassVar[set[str]] = {"running", "succeeded", "failed"}

    @staticmethod
    def list_runs_sync(
        order_by: str = "start_time",
        ascending: bool = False,
        status: str | None = None,
        task_name: str | None = None,
    ) -> list[RunRecord]:
        """List top-level runs (action_name='a0') with configurable sort and filter."""
        if order_by not in RunStore._SORTABLE_COLUMNS:
            order_by = "start_time"
        direction = "ASC" if ascending else "DESC"
        sql_order = "(end_time - start_time)" if order_by == "duration" else order_by

        where = "action_name='a0'"
        params: list[str] = []
        if status and status in RunStore._VALID_STATUSES:
            where += " AND status=?"
            params.append(status)
        if task_name:
            where += " AND task_name LIKE ?"
            params.append(f"%{task_name}%")

        conn = LocalDB.get_sync()
        cursor = conn.execute(
            f"""
            SELECT
                r.*,
                COALESCE(
                    (
                        SELECT MAX(
                            CASE
                                WHEN COALESCE(sr.attempt_count, 0) > 0 THEN sr.attempt_count
                                ELSE 1
                            END
                        )
                        FROM runs sr
                        WHERE sr.run_name = r.run_name
                    ),
                    1
                ) AS max_attempts_used,
                COALESCE(
                    (
                        SELECT COUNT(*)
                        FROM runs sr
                        WHERE sr.run_name = r.run_name
                          AND COALESCE(sr.attempt_count, 0) > 1
                    ),
                    0
                ) AS retried_actions
            FROM runs r
            WHERE {where}
            ORDER BY {sql_order} {direction}
            """,
            params,
        )
        column_names = [d[0] for d in cursor.description]
        records: list[RunRecord] = []
        for row in cursor.fetchall():
            row_map = dict(zip(column_names, row))
            base = RunStore._row_to_record(row_map)
            base.max_attempts_used = int(row_map.get("max_attempts_used") or 1)
            base.retried_actions = int(row_map.get("retried_actions") or 0)
            records.append(base)
        return records

    @staticmethod
    def list_actions_for_run_sync(run_name: str) -> list[RunRecord]:
        """List all actions for a given run."""
        conn = LocalDB.get_sync()
        cursor = conn.execute(
            "SELECT * FROM runs WHERE run_name=? ORDER BY start_time ASC",
            (run_name,),
        )
        column_names = [d[0] for d in cursor.description]
        return [RunStore._row_to_record(dict(zip(column_names, row))) for row in cursor.fetchall()]

    @staticmethod
    def get_action_sync(run_name: str, action_name: str) -> RunRecord | None:
        conn = LocalDB.get_sync()
        cursor = conn.execute(
            "SELECT * FROM runs WHERE run_name=? AND action_name=?",
            (run_name, action_name),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        column_names = [d[0] for d in cursor.description]
        return RunStore._row_to_record(dict(zip(column_names, row)))

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
    def _row_to_record(row: dict) -> RunRecord:
        """Build a ``RunRecord`` from a column-name → value mapping. Reading
        by name (instead of positional indexing) is required because old
        DBs accumulated migrated columns at the tail in a different order
        than freshly-created DBs from the current ``_RUNS_DDL``."""
        return RunRecord(
            run_name=row["run_name"],
            action_name=row["action_name"],
            task_name=row.get("task_name"),
            status=row.get("status") or "running",
            inputs=row.get("inputs"),
            outputs=row.get("outputs"),
            error=row.get("error"),
            start_time=row.get("start_time"),
            end_time=row.get("end_time"),
            parent_id=row.get("parent_id"),
            short_name=row.get("short_name"),
            output_path=row.get("output_path"),
            cache_enabled=bool(row.get("cache_enabled")),
            cache_hit=bool(row.get("cache_hit")),
            disable_run_cache=bool(row.get("disable_run_cache")),
            has_report=bool(row.get("has_report")),
            context=row.get("context"),
            group_name=row.get("group_name"),
            log_links=row.get("log_links"),
            attempt_count=int(row.get("attempt_count") or 0),
            attempts_json=row.get("attempts_json"),
        )
