import { useEffect, useMemo, useState } from "react";

type RunInput = Record<string, unknown>;

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
  input: RunInput;
  output: Record<string, unknown> | null;
  tasks: TaskResult[];
  workflow_module?: string | null;
  workflow_name?: string | null;
  raw_args?: Record<string, unknown> | null;
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

const parseInput = (text: string): RunInput => {
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed as RunInput;
    }
  } catch {
    // Fall through to best-effort x_list parsing
  }
  const x_list = text
    .split(/[\s,]+/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => Number(chunk))
    .filter((value) => !Number.isNaN(value));
  return { x_list };
};

export default function Home() {
  const [inputText, setInputText] = useState("{\"x_list\": [1,2,3,3,3,3,3,3,3,3,3,3]}");
  const [runs, setRuns] = useState<RunResult[]>([]);
  const [activeRun, setActiveRun] = useState<RunResult | null>(null);
  const [selectedTaskIndex, setSelectedTaskIndex] = useState<number>(0);
  const [isLaunching, setIsLaunching] = useState(false);
  const [activeTab, setActiveTab] = useState<"summary" | "reports">("summary");

  const selectedTask = useMemo(() => {
    if (!activeRun?.tasks?.length) return null;
    return activeRun.tasks[selectedTaskIndex] ?? null;
  }, [activeRun, selectedTaskIndex]);

  const refreshRuns = async () => {
    const response = await fetch(`${API_BASE}/api/runs`);
    if (!response.ok) return;
    const data: RunResult[] = await response.json();
    setRuns(data);
  };

  const refreshRun = async (runId: string) => {
    const response = await fetch(`${API_BASE}/api/runs/${runId}`);
    if (!response.ok) return;
    const data: RunResult = await response.json();
    setActiveRun(data);
    return data;
  };

  const handleRun = async () => {
    const payload: RunInput = parseInput(inputText);
    setIsLaunching(true);
    try {
      const response = await fetch(`${API_BASE}/api/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) return;
      const data: RunResult = await response.json();
      setActiveRun(data);
      setSelectedTaskIndex(0);
      refreshRuns();
    } finally {
      setIsLaunching(false);
    }
  };


  useEffect(() => {
    refreshRuns();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      refreshRuns();
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!activeRun) return;
    let timer: NodeJS.Timeout | null = null;
    const poll = async () => {
      const data = await refreshRun(activeRun.run_id);
      if (!data || data.status !== "running") return;
      timer = setTimeout(poll, 900);
    };
    poll();
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [activeRun?.run_id]);

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">F</span>
          <div>
            <div className="brand-title">Flyte Local UI</div>
            <div className="brand-sub">Development • flytesnacks</div>
          </div>
        </div>
        <div className="search">
          <input placeholder="Search" />
        </div>
        <div className="user">KS</div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="panel-header">
            <div className="panel-title">Runs</div>
            <div className="panel-count">{runs.length}</div>
          </div>
          <div className="panel-body scroll">
            {runs.length === 0 && <div className="empty">No runs yet</div>}
            {runs.map((run) => (
              <button
                key={run.run_id}
                className={`run-item ${activeRun?.run_id === run.run_id ? "active" : ""}`}
                onClick={() => {
                  setActiveRun(run);
                  setSelectedTaskIndex(0);
                }}
              >
                <div className="run-meta">
                  <span className={`status ${run.status}`}>{run.status}</span>
                  <span className="run-id">{run.run_id}</span>
                </div>
                <div className="run-sub">{formatTime(run.start_time)}</div>
                <div className="run-duration">{formatMs(run.duration_ms)}</div>
              </button>
            ))}
          </div>
        </aside>

        <aside className="sidebar">
          <div className="panel-header">
            <div className="panel-title">Tasks</div>
            <div className="panel-count">{activeRun?.tasks.length ?? 0}</div>
          </div>
          <div className="panel-body scroll">
            {activeRun?.tasks.map((task, idx) => (
              <button
                key={`${task.name}-${task.index}-${idx}`}
                className={`task-item ${selectedTaskIndex === idx ? "active" : ""}`}
                onClick={() => setSelectedTaskIndex(idx)}
              >
                <div className="task-title">{task.name} {task.index}</div>
                <div className="task-meta">
                  <span className={`pill ${task.status}`}>{task.status}</span>
                  <span>{formatMs(task.duration_ms)}</span>
                </div>
                <div className="task-progress">
                  <span style={{ width: `${Math.min(100, (task.duration_ms / 900) * 100)}%` }} />
                </div>
              </button>
            ))}
          </div>
        </aside>

        <main className="content">
          <div className="run-header">
            <div>
              <div className="run-title">main</div>
              <div className="run-info">Run ID: {activeRun?.run_id ?? "-"}</div>
            </div>
            <div className="run-stats">
              <div>
                <div className="label">Duration</div>
                <div className="value">{formatMs(activeRun?.duration_ms)}</div>
              </div>
              <div>
                <div className="label">Start Time</div>
                <div className="value">{formatTime(activeRun?.start_time)}</div>
              </div>
              <div>
                <div className="label">Trigger</div>
                <div className="value">local</div>
              </div>
              <div>
                <div className="label">Owned By</div>
                <div className="value">Kevin Su</div>
              </div>
            </div>
            <div className="run-note">Run workflows locally with the Flyte CLI to see them here.</div>
          </div>

          <div className="tab-bar">
            <button
              className={`tab ${activeTab === "summary" ? "active" : ""}`}
              onClick={() => setActiveTab("summary")}
            >
              Summary
            </button>
            <button
              className={`tab ${activeTab === "reports" ? "active" : ""}`}
              onClick={() => setActiveTab("reports")}
            >
              Reports
            </button>
          </div>

          {activeTab === "summary" ? (
            <>
              <div className="card">
                <div className="card-title">Workflow Summary</div>
                <div className="summary-row">
                  <div>
                    <div className="summary-label">Status</div>
                    <div className={`summary-value ${activeRun?.status ?? "pending"}`}>{activeRun?.status ?? "-"}</div>
                  </div>
                  <div>
                    <div className="summary-label">Total Tasks</div>
                    <div className="summary-value">{activeRun?.tasks.length ?? 0}</div>
                  </div>
                  <div>
                    <div className="summary-label">Completed</div>
                    <div className="summary-value">
                      {activeRun?.tasks.filter((task) => task.status === "completed").length ?? 0}
                    </div>
                  </div>
                  <div>
                    <div className="summary-label">Failed</div>
                    <div className="summary-value">
                      {activeRun?.tasks.filter((task) => task.status === "failed").length ?? 0}
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid">
                <div className="card">
                  <div className="card-title">Task Details</div>
                  {selectedTask ? (
                    <div className="detail-list">
                      <div className="detail-row">
                        <span>Task</span>
                        <span>{selectedTask.name} {selectedTask.index}</span>
                      </div>
                      <div className="detail-row">
                        <span>Status</span>
                        <span className={`pill ${selectedTask.status}`}>{selectedTask.status}</span>
                      </div>
                      <div className="detail-row">
                        <span>Duration</span>
                        <span>{formatMs(selectedTask.duration_ms)}</span>
                      </div>
                      <div className="detail-row">
                        <span>Start</span>
                        <span>{formatTime(selectedTask.start_time)}</span>
                      </div>
                      <div className="detail-row">
                        <span>End</span>
                        <span>{formatTime(selectedTask.end_time)}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="empty">No task selected</div>
                  )}
                </div>

                <div className="card">
                  <div className="card-title">Input</div>
                  <div className="input-box">
                    <textarea value={inputText} onChange={(event) => setInputText(event.target.value)} />
                    <div className="hint">Provide a comma or space-separated list of numbers.</div>
                  </div>
                </div>
              </div>

              <div className="grid">
                <div className="card">
                  <div className="card-title">Output</div>
                  <pre className="code-block">
                    {activeRun?.output ? JSON.stringify(activeRun.output, null, 2) : "Run the workflow to see output."}
                  </pre>
                </div>
                <div className="card">
                  <div className="card-title">Input Snapshot</div>
                  <pre className="code-block">
                    {activeRun?.input ? JSON.stringify(activeRun.input, null, 2) : "Waiting for input."}
                  </pre>
                </div>
              </div>

              <div className="card">
                <div className="card-title">Task Logs</div>
                <pre className="code-block">
                  {selectedTask?.log
                    ? selectedTask.log
                    : "Select a task to view logs."}
                </pre>
              </div>
            </>
          ) : (
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
    </div>
  );
}
