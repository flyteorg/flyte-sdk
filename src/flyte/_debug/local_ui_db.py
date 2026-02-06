from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ENV_ENABLE = "FLYTE_UI_ENABLE"
ENV_DB_PATH = "FLYTE_UI_DB_PATH"

_db_lock = threading.Lock()
_db: Optional[sqlite3.Connection] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "1").lower() not in {"0", "false", "no"}


def _resolve_db_path() -> Optional[Path]:
    env_path = os.environ.get(ENV_DB_PATH)
    if env_path:
        return Path(env_path).expanduser().resolve()

    cwd = Path.cwd()
    repo_db = cwd / "ui" / "backend" / "local_runs.db"
    if repo_db.parent.exists():
        return repo_db.resolve()

    fallback = Path("~/.flyte/local_ui_runs.db").expanduser()
    return fallback


def _get_db_path() -> Optional[Path]:
    if not _is_enabled():
        return None
    return _resolve_db_path()


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        db_path = _get_db_path()
        if db_path is None:
            raise RuntimeError("Local UI DB path is not configured.")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(db_path, check_same_thread=False)
        _db.row_factory = sqlite3.Row
    return _db


def _init_db() -> None:
    db_path = _get_db_path()
    if db_path is None:
        return
    db = _get_db()
    with _db_lock:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_ms REAL,
                input_json TEXT NOT NULL,
                output_json TEXT,
                workflow_module TEXT,
                workflow_name TEXT,
                raw_args_json TEXT
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                action_name TEXT,
                task_index INTEGER NOT NULL,
                input REAL NOT NULL,
                output REAL,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                log_text TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id);
            """
        )
        cols = {row["name"] for row in db.execute("PRAGMA table_info(runs)").fetchall()}
        if "workflow_module" not in cols:
            db.execute("ALTER TABLE runs ADD COLUMN workflow_module TEXT")
        if "workflow_name" not in cols:
            db.execute("ALTER TABLE runs ADD COLUMN workflow_name TEXT")
        if "raw_args_json" not in cols:
            db.execute("ALTER TABLE runs ADD COLUMN raw_args_json TEXT")
        cols = {row["name"] for row in db.execute("PRAGMA table_info(tasks)").fetchall()}
        if "report_html" not in cols:
            db.execute("ALTER TABLE tasks ADD COLUMN report_html TEXT")
        if "log_text" not in cols:
            db.execute("ALTER TABLE tasks ADD COLUMN log_text TEXT")
        if "action_name" not in cols:
            db.execute("ALTER TABLE tasks ADD COLUMN action_name TEXT")
        db.commit()


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=repr)


def ensure_run(
    run_id: str,
    input_payload: Dict[str, Any],
    workflow_module: Optional[str] = None,
    workflow_name: Optional[str] = None,
    raw_args: Optional[Dict[str, Any]] = None,
) -> None:
    if _get_db_path() is None:
        return
    _init_db()
    db = _get_db()
    with _db_lock:
        row = db.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            db.execute(
                """
                INSERT INTO runs (
                    run_id, status, start_time, end_time, duration_ms, input_json, output_json,
                    workflow_module, workflow_name, raw_args_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    "running",
                    _utc_now_iso(),
                    None,
                    None,
                    _json_dumps(input_payload),
                    None,
                    workflow_module,
                    workflow_name,
                    _json_dumps(raw_args or {}),
                ),
            )
            db.commit()


def update_run(run_id: str, status: str, duration_ms: float, output_payload: Dict[str, Any]) -> None:
    if _get_db_path() is None:
        return
    _init_db()
    db = _get_db()
    with _db_lock:
        row = db.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            db.execute(
                """
                INSERT INTO runs (run_id, status, start_time, end_time, duration_ms, input_json, output_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, status, _utc_now_iso(), _utc_now_iso(), duration_ms, "{}", _json_dumps(output_payload)),
            )
        else:
            db.execute(
                "UPDATE runs SET status = ?, end_time = ?, duration_ms = ?, output_json = ? WHERE run_id = ?",
                (status, _utc_now_iso(), duration_ms, _json_dumps(output_payload), run_id),
            )
        db.commit()


def record_task(
    run_id: str,
    name: str,
    input_value: float,
    output_value: Optional[float],
    status: str,
    start_time: str,
    end_time: str,
    duration_ms: float,
    log_text: str,
    report_html: Optional[str],
) -> None:
    if _get_db_path() is None:
        return
    _init_db()
    db = _get_db()
    with _db_lock:
        row = db.execute(
            "SELECT COUNT(*) as cnt FROM tasks WHERE run_id = ? AND name = ?",
            (run_id, name),
        ).fetchone()
        task_index = int(row["cnt"]) if row else 0
        db.execute(
            """
            INSERT INTO tasks (
                run_id, name, task_index, input, output, status, start_time, end_time, duration_ms, log_text, report_html
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                name,
                task_index,
                input_value,
                output_value,
                status,
                start_time,
                end_time,
                duration_ms,
                log_text,
                report_html,
            ),
        )
        db.commit()


def record_task_start(
    run_id: str,
    name: str,
    action_name: str,
    input_value: float,
    start_time: str,
) -> int:
    if _get_db_path() is None:
        return -1
    _init_db()
    db = _get_db()
    with _db_lock:
        row = db.execute(
            "SELECT COUNT(*) as cnt FROM tasks WHERE run_id = ? AND name = ?",
            (run_id, name),
        ).fetchone()
        task_index = int(row["cnt"]) if row else 0
        cur = db.execute(
            """
            INSERT INTO tasks (
                run_id, name, action_name, task_index, input, output, status, start_time, end_time, duration_ms, log_text, report_html
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                name,
                action_name,
                task_index,
                input_value,
                None,
                "running",
                start_time,
                start_time,
                0.0,
                "",
                None,
            ),
        )
        db.commit()
        return int(cur.lastrowid)


def record_task_finish(
    row_id: int,
    output_value: Optional[float],
    status: str,
    end_time: str,
    duration_ms: float,
    log_text: str,
    report_html: Optional[str],
) -> None:
    if _get_db_path() is None or row_id < 0:
        return
    _init_db()
    db = _get_db()
    with _db_lock:
        db.execute(
            """
            UPDATE tasks
            SET output = ?, status = ?, end_time = ?, duration_ms = ?, log_text = ?, report_html = ?
            WHERE id = ?
            """,
            (output_value, status, end_time, duration_ms, log_text, report_html, row_id),
        )
        db.commit()


def coerce_inputs(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Dict[str, Any]:
    try:
        import inspect

        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        return dict(bound.arguments)
    except Exception:
        if kwargs:
            return dict(kwargs)
        return {f"arg{idx}": value for idx, value in enumerate(args)}


def get_run_id_from_context() -> Optional[str]:
    try:
        from flyte._context import internal_ctx
    except Exception:
        return None
    ctx = internal_ctx()
    tctx = ctx.data.task_context if ctx else None
    if not tctx or not tctx.run_base_dir:
        return None
    return Path(tctx.run_base_dir).name


def maybe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_report_html(output_path: str) -> Optional[str]:
    try:
        from flyte._internal.runtime import io
    except Exception:
        return None
    report_path = io.report_path(output_path)
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except Exception:
        return None


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
