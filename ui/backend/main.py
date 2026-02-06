from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.basics import hello
app = FastAPI(title="Flyte Local UI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunInput(BaseModel):
    root: Dict[str, Any]

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        return self.root


class TaskResult(BaseModel):
    name: str
    index: int
    input: float
    output: Optional[float]
    status: str
    start_time: str
    end_time: str
    duration_ms: float
    log: str
    report_html: Optional[str]


class RunResult(BaseModel):
    run_id: str
    status: str
    start_time: str
    end_time: Optional[str]
    duration_ms: Optional[float]
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]]
    tasks: List[TaskResult]
    workflow_module: Optional[str] = None
    workflow_name: Optional[str] = None
    raw_args: Optional[Dict[str, Any]] = None


def _resolve_db_path() -> Path:
    env_path = os.environ.get("FLYTE_UI_DB_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    repo_db = repo_root / "ui" / "backend" / "local_runs.db"
    if repo_db.parent.exists():
        return repo_db
    return Path("~/.flyte/local_ui_runs.db").expanduser()


DB_PATH = _resolve_db_path()
_db_lock = threading.Lock()
_db: Optional[sqlite3.Connection] = None


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db.row_factory = sqlite3.Row
    return _db


def _init_db() -> None:
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
                task_index INTEGER NOT NULL,
                input REAL NOT NULL,
                output REAL,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                duration_ms REAL NOT NULL,
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
        if "log_text" not in cols:
            db.execute("ALTER TABLE tasks ADD COLUMN log_text TEXT")
        if "report_html" not in cols:
            db.execute("ALTER TABLE tasks ADD COLUMN report_html TEXT")
        db.commit()


@app.on_event("startup")
def _startup() -> None:
    _init_db()


def _simulate_task(value: float) -> float:
    # Use the real hello.py task function locally.
    return float(hello.fn(int(value)))


def _execute_run(run_id: str, run_input: RunInput) -> None:
    outputs: List[float] = []
    tasks: List[TaskResult] = []
    run_start = time.perf_counter()
    input_payload = run_input.model_dump()
    x_list = input_payload.get("x_list", [])
    if not isinstance(x_list, list):
        x_list = []
    if len(x_list) < 10:
        end_time = _utc_now_iso()
        duration_ms = (time.perf_counter() - run_start) * 1000
        output_json = json.dumps(
            {
                "error": "x_list doesn't have a larger enough sample size",
                "count": len(x_list),
            }
        )
        with _db_lock:
            db = _get_db()
            db.execute(
                "UPDATE runs SET status = ?, end_time = ?, duration_ms = ?, output_json = ? WHERE run_id = ?",
                ("failed", end_time, duration_ms, output_json, run_id),
            )
            db.commit()
        return
    for idx, value in enumerate(x_list):
        start_time = _utc_now_iso()
        start_perf = time.perf_counter()
        status = "completed"
        try:
            output = _simulate_task(value)
        except Exception:
            status = "failed"
            output = float("nan")
        end_time = _utc_now_iso()
        duration_ms = (time.perf_counter() - start_perf) * 1000
        outputs.append(output)
        log_lines = [
            f"[{start_time}] task fn[{idx}] start input={value}",
            f"[{end_time}] task fn[{idx}] end output={output} status={status} duration_ms={duration_ms:.0f}",
        ]
        log_text = "\n".join(log_lines)
        task = TaskResult(
            name="fn",
            index=idx,
            input=value,
            output=output,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            log=log_text,
            report_html=None,
        )
        tasks.append(task)
        with _db_lock:
            db = _get_db()
            db.execute(
                """
                INSERT INTO tasks (
                    run_id, name, task_index, input, output, status, start_time, end_time, duration_ms, log_text, report_html
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task.name,
                    task.index,
                    task.input,
                    task.output,
                    task.status,
                    task.start_time,
                    task.end_time,
                    task.duration_ms,
                    task.log,
                    task.report_html,
                ),
            )
            db.commit()

    end_time = _utc_now_iso()
    duration_ms = (time.perf_counter() - run_start) * 1000
    status = "completed" if all(t.status == "completed" for t in tasks) else "failed"
    output_json = json.dumps(
        {"y_list": outputs, "y_mean": sum(outputs) / len(outputs) if outputs else None}
    )
    with _db_lock:
        db = _get_db()
        db.execute(
            "UPDATE runs SET status = ?, end_time = ?, duration_ms = ?, output_json = ? WHERE run_id = ?",
            (status, end_time, duration_ms, output_json, run_id),
        )
        db.commit()


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunResult)
def create_run(run_input: RunInput, background_tasks: BackgroundTasks) -> RunResult:
    run_id = uuid.uuid4().hex[:10]
    start_time = _utc_now_iso()
    input_payload = run_input.model_dump()
    with _db_lock:
        db = _get_db()
        db.execute(
            """
            INSERT INTO runs (run_id, status, start_time, end_time, duration_ms, input_json, output_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, "running", start_time, None, None, json.dumps(input_payload), None),
        )
        db.commit()
    background_tasks.add_task(_execute_run, run_id, run_input)
    return RunResult(
        run_id=run_id,
        status="running",
        start_time=start_time,
        end_time=None,
        duration_ms=None,
        input=input_payload,
        output=None,
        tasks=[],
    )


@app.get("/api/runs", response_model=List[RunResult])
def list_runs() -> List[RunResult]:
    with _db_lock:
        db = _get_db()
        rows = db.execute(
            "SELECT run_id, status, start_time, end_time, duration_ms, input_json, output_json, workflow_module, workflow_name, raw_args_json FROM runs ORDER BY start_time DESC"
        ).fetchall()
    results: List[RunResult] = []
    for row in rows:
        input_data = json.loads(row["input_json"]) if row["input_json"] else {}
        output_data = json.loads(row["output_json"]) if row["output_json"] else None
        raw_args = json.loads(row["raw_args_json"]) if row["raw_args_json"] else None
        results.append(
            RunResult(
                run_id=row["run_id"],
                status=row["status"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                duration_ms=row["duration_ms"],
                input=input_data,
                output=output_data,
                tasks=[],
                workflow_module=row["workflow_module"],
                workflow_name=row["workflow_name"],
                raw_args=raw_args,
            )
        )
    return results


@app.get("/api/runs/{run_id}", response_model=RunResult)
def get_run(run_id: str) -> RunResult:
    with _db_lock:
        db = _get_db()
        row = db.execute(
            "SELECT run_id, status, start_time, end_time, duration_ms, input_json, output_json, workflow_module, workflow_name, raw_args_json FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        task_rows = db.execute(
            """
            SELECT name, task_index, input, output, status, start_time, end_time, duration_ms, log_text, report_html
            FROM tasks
            WHERE run_id = ?
            ORDER BY task_index ASC
            """,
            (run_id,),
        ).fetchall()
    input_data = json.loads(row["input_json"]) if row["input_json"] else {}
    output_data = json.loads(row["output_json"]) if row["output_json"] else None
    raw_args = json.loads(row["raw_args_json"]) if row["raw_args_json"] else None
    tasks = [
        TaskResult(
            name=task["name"],
            index=task["task_index"],
            input=task["input"],
            output=task["output"],
            status=task["status"],
            start_time=task["start_time"],
            end_time=task["end_time"],
            duration_ms=task["duration_ms"],
            log=task["log_text"] or "",
            report_html=task["report_html"],
        )
        for task in task_rows
    ]
    return RunResult(
        run_id=row["run_id"],
        status=row["status"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        duration_ms=row["duration_ms"],
        input=input_data,
        output=output_data,
        tasks=tasks,
        workflow_module=row["workflow_module"],
        workflow_name=row["workflow_name"],
        raw_args=raw_args,
    )

