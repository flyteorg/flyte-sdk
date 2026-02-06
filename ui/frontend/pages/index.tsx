import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type RunInput = Record<string, unknown>;

type RunResult = {
  run_id: string;
  status: string;
  start_time: string;
  end_time: string | null;
  duration_ms: number | null;
  input: RunInput;
  output: Record<string, unknown> | null;
  tasks: unknown[];
  workflow_module?: string | null;
  workflow_name?: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

const formatMs = (ms?: number | null) => {
  if (ms === undefined || ms === null) return "-";
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const formatTime = (iso?: string | null) => {
  if (!iso) return "-";
  const date = new Date(iso);
  return date.toLocaleString();
};

const statusDotClass = (status?: string) => {
  if (status === "failed") return "dot failed";
  if (status === "running") return "dot running";
  return "dot success";
};

export default function RunsList() {
  const [runs, setRuns] = useState<RunResult[]>([]);
  const [query, setQuery] = useState("");

  const refreshRuns = async () => {
    const response = await fetch(`${API_BASE}/api/runs`);
    if (!response.ok) return;
    const data: RunResult[] = await response.json();
    setRuns(data);
  };

  useEffect(() => {
    refreshRuns();
    const interval = setInterval(() => refreshRuns(), 2000);
    return () => clearInterval(interval);
  }, []);

  const filtered = useMemo(() => {
    if (!query) return runs;
    const q = query.toLowerCase();
    return runs.filter((run) =>
      [run.run_id, run.workflow_name, run.workflow_module]
        .filter(Boolean)
        .some((value) => value!.toLowerCase().includes(q))
    );
  }, [query, runs]);

  return (
    <div className="app-shell">
      <aside className="nav-rail">
        <div className="nav-logo">F</div>
        <nav className="nav-items">
          <button className="nav-item active">
            <span className="nav-icon">✦</span>
            Runs
          </button>
        </nav>
        <div className="nav-footer">Settings</div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-btn">☰</button>
            <div className="context-pill">Development</div>
            <div className="context-pill">Flyte SDK</div>
          </div>
          <div className="topbar-center">
            <div className="search">
              <input
                placeholder="Search runs"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
          </div>
          <div className="topbar-right">
            <div className="avatar">KS</div>
          </div>
        </header>

        <section className="content-area">
          <div className="page-header">
            <div>
              <div className="page-title">Runs</div>
              <div className="page-sub">Runs in the last 7 days</div>
            </div>
          </div>

          <div className="table">
            <div className="table-row header">
              <div>Run</div>
              <div>Trigger</div>
              <div>Duration</div>
              <div>Start time</div>
              <div>End time</div>
              <div>Owner</div>
            </div>
            {filtered.map((run) => (
              <Link
                key={run.run_id}
                href={`/runs/${run.run_id}`}
                className="table-row"
              >
                <div className="run-cell">
                  <span className={statusDotClass(run.status)} />
                  <div>
                    <div className="run-name">{run.workflow_name ?? "main"}</div>
                    <div className="run-id">Run ID: {run.run_id}</div>
                  </div>
                </div>
                <div className="muted">-</div>
                <div>{formatMs(run.duration_ms)}</div>
                <div className="muted">{formatTime(run.start_time)}</div>
                <div className="muted">{formatTime(run.end_time)}</div>
                <div className="owner">KS</div>
              </Link>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
