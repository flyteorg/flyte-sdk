import Link from "next/link";
import { useRouter } from "next/router";
import { useEffect, useMemo, useState } from "react";

type TaskResult = {
  name: string;
  index: number;
  input: number;
  output: number | null;
  status: string;
  start_time: string;
  end_time: string;
  duration_ms: number;
  log: string;
  report_html?: string | null;
};

type RunResult = {
  run_id: string;
  status: string;
  start_time: string;
  end_time: string | null;
  duration_ms: number | null;
  input: Record<string, unknown>;
  output: Record<string, unknown> | null;
  tasks: TaskResult[];
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

export default function RunDetail() {
  const router = useRouter();
  const { run_id } = router.query;
  const [run, setRun] = useState<RunResult | null>(null);
  const [selectedTaskIndex, setSelectedTaskIndex] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<"summary" | "reports" | "logs">("summary");

  const selectedTask = useMemo(() => {
    if (!run?.tasks?.length) return null;
    return run.tasks[selectedTaskIndex] ?? null;
  }, [run, selectedTaskIndex]);

  const refresh = async () => {
    if (!run_id || typeof run_id !== "string") return;
    const response = await fetch(`${API_BASE}/api/runs/${run_id}`);
    if (!response.ok) return;
    const data: RunResult = await response.json();
    setRun(data);
  };

  useEffect(() => {
    refresh();
    const interval = setInterval(() => refresh(), 2000);
    return () => clearInterval(interval);
  }, [run_id]);

  return (
    <div className="app-shell">
      <aside className="nav-rail">
        <div className="nav-logo">F</div>
        <nav className="nav-items">
          <Link className="nav-item" href="/">
            <span className="nav-icon">✦</span>
            Runs
          </Link>
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
              <input placeholder="Search" />
            </div>
          </div>
          <div className="topbar-right">
            <div className="avatar">KS</div>
          </div>
        </header>

        <section className="detail-shell">
          <div className="detail-header">
            <div>
              <div className="detail-title">{run?.workflow_name ?? "main"}</div>
              <div className="detail-sub">Run ID: {run?.run_id ?? "-"}</div>
            </div>
            <div className="detail-metrics">
              <div>
                <div className="label">Duration</div>
                <div className="value">{formatMs(run?.duration_ms)}</div>
              </div>
              <div>
                <div className="label">Start Time</div>
                <div className="value">{formatTime(run?.start_time)}</div>
              </div>
              <div>
                <div className="label">Trigger</div>
                <div className="value">-</div>
              </div>
              <div>
                <div className="label">Owned By</div>
                <div className="value">Flyte SDK</div>
              </div>
            </div>
          </div>

          <div className="detail-grid">
            <aside className="sidebar">
              <div className="panel-header">
                <div className="panel-title">Actions</div>
                <div className="panel-count">{run?.tasks.length ?? 0}</div>
              </div>
              <div className="panel-body scroll">
                {run?.tasks.map((task, idx) => (
                  <button
                    key={`${task.name}-${task.index}-${idx}`}
                    className={`task-item ${selectedTaskIndex === idx ? "active" : ""}`}
                    onClick={() => setSelectedTaskIndex(idx)}
                  >
                    <div className="task-row">
                      <span className={`task-spinner ${task.status.toLowerCase()}`} />
                      <div className="task-main">
                        <div className="task-title">{task.name}</div>
                        <div className="task-meta">
                          <span className={`pill ${task.status.toLowerCase()}`}>{task.status}</span>
                          <span>{formatMs(task.duration_ms)}</span>
                        </div>
                      </div>
                      <div className="task-timeline">
                        <span style={{ width: `${Math.min(100, (task.duration_ms / 900) * 100)}%` }} />
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </aside>

            <main className="content">
              <div className="tab-bar">
                <button
                  className={`tab ${activeTab === "summary" ? "active" : ""}`}
                  onClick={() => setActiveTab("summary")}
                >
                  Summary
                </button>
                <button
                  className={`tab ${activeTab === "logs" ? "active" : ""}`}
                  onClick={() => setActiveTab("logs")}
                >
                  Logs
                </button>
                <button
                  className={`tab ${activeTab === "reports" ? "active" : ""}`}
                  onClick={() => setActiveTab("reports")}
                >
                  Reports
                </button>
              </div>

              {activeTab === "summary" && (
                <>
                  <div className="card">
                    <div className="card-title">Summary</div>
                    <div className="summary-row">
                      <div>
                        <div className="summary-label">Status</div>
                        <div className={`summary-value ${run?.status ?? "pending"}`}>{run?.status ?? "-"}</div>
                      </div>
                      <div>
                        <div className="summary-label">Total Tasks</div>
                        <div className="summary-value">{run?.tasks.length ?? 0}</div>
                      </div>
                      <div>
                        <div className="summary-label">Completed</div>
                        <div className="summary-value">
                          {run?.tasks.filter((task) => task.status === "completed").length ?? 0}
                        </div>
                      </div>
                      <div>
                        <div className="summary-label">Failed</div>
                        <div className="summary-value">
                          {run?.tasks.filter((task) => task.status === "failed").length ?? 0}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="grid">
                    <div className="card">
                      <div className="card-title">Output</div>
                      <pre className="code-block">
                        {run?.output ? JSON.stringify(run.output, null, 2) : "No output."}
                      </pre>
                    </div>
                    <div className="card">
                      <div className="card-title">Input</div>
                      <pre className="code-block">
                        {run?.input ? JSON.stringify(run.input, null, 2) : "No input."}
                      </pre>
                    </div>
                  </div>
                </>
              )}

              {activeTab === "logs" && (
                <div className="card">
                  <div className="card-title">Task Logs</div>
                  <pre className="code-block">
                    {selectedTask?.log ? selectedTask.log : "Select an action to view logs."}
                  </pre>
                </div>
              )}

              {activeTab === "reports" && (
                <div className="card">
                  <div className="card-title">Task Report</div>
                  {selectedTask?.report_html ? (
                    <iframe
                      className="report-frame"
                      title="Flyte Report"
                      sandbox="allow-same-origin allow-scripts"
                      srcDoc={selectedTask.report_html}
                    />
                  ) : (
                    <div className="empty">No report found for the selected task.</div>
                  )}
                </div>
              )}
            </main>
          </div>
        </section>
      </div>
    </div>
  );
}
