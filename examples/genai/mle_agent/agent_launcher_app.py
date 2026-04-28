# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "flyte",
#   "fastapi",
#   "uvicorn",
#   "python-multipart",
# ]
# ///

"""Agent Launcher App — A Union.ai-themed console for launching MLE agents.

Provides a web UI where users can:
1. Select an agent (MLE Tool Builder or MLE Orchestrator)
2. Enter a prompt describing the task
3. Launch the agent and monitor execution status with live polling
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette import status

import flyte
import flyte.app
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment, FastAPIPassthroughAuthMiddleware
from flyte.io import File

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent registry — add new agents here
# ---------------------------------------------------------------------------

AGENTS = [
    {
        "id": "mle_tool_builder",
        "name": "MLE Tool Builder",
        "task_name": "mle-tool-builder.mle_tool_builder_agent",
        "description": (
            "Builds its own tools via code sandbox. Generates Python code,"
            " executes it in an isolated sandbox, and iteratively fixes errors."
        ),
        "icon": "hammer",
        "requires_data": True,
    },
    {
        "id": "mle_orchestrator",
        "name": "MLE Orchestrator",
        "task_name": "mle-orchestrator.mle_orchestrator_agent",
        "description": (
            "Builds orchestration code using pre-defined tools."
            " Generates code that calls tool tasks and iteratively fixes errors."
        ),
        "icon": "workflow",
        "requires_data": True,
    },
]

# ---------------------------------------------------------------------------
# HTML page — Union.ai black & gold theme
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Launcher</title>
<meta name="theme-color" content="#050505">
<style>
  :root {
    --bg: #050505;
    --panel: rgba(17, 17, 17, 0.92);
    --panel-strong: #131313;
    --border: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.14);
    --text: #f4f4f5;
    --text-soft: #a1a1aa;
    --text-muted: #71717a;
    --accent: #fcb51f;
    --accent-soft: rgba(252, 181, 31, 0.14);
    --accent-strong: #ffd46b;
    --success: #65bd15;
    --danger: #f87171;
    --shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
    --radius-xl: 28px;
    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;
    color-scheme: dark;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    min-height: 100%;
    background:
      radial-gradient(circle at top left, rgba(252, 181, 31, 0.12), transparent 26%),
      radial-gradient(circle at top right, rgba(252, 181, 31, 0.06), transparent 22%),
      linear-gradient(180deg, #090909 0%, #050505 48%, #020202 100%);
  }

  body {
    font-family: "Avenir Next", "SF Pro Display", "Segoe UI Variable", "Segoe UI", sans-serif;
    color: var(--text);
    padding: 28px 18px 40px;
    position: relative;
    overflow-x: hidden;
  }

  body::before {
    content: "";
    position: fixed;
    top: -120px;
    left: -90px;
    width: 280px;
    height: 280px;
    border-radius: 999px;
    pointer-events: none;
    filter: blur(100px);
    opacity: 0.4;
    background: rgba(252, 181, 31, 0.2);
    z-index: 0;
  }

  .shell {
    position: relative;
    z-index: 1;
    max-width: 960px;
    margin: 0 auto;
  }

  /* ---- Hero ---- */
  .hero {
    background: linear-gradient(180deg, rgba(27, 27, 27, 0.96) 0%, rgba(12, 12, 12, 0.96) 100%);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-xl);
    padding: 28px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
    margin-bottom: 22px;
    overflow: hidden;
    position: relative;
  }

  .hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
      linear-gradient(125deg, rgba(252, 181, 31, 0.08), transparent 38%),
      linear-gradient(180deg, transparent, rgba(255, 255, 255, 0.02));
    pointer-events: none;
  }

  .brand-block {
    display: flex;
    gap: 18px;
    align-items: flex-start;
    position: relative;
  }

  .brand-mark {
    width: 64px;
    height: 64px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(252, 181, 31, 0.18), rgba(252, 181, 31, 0.06));
    border: 1px solid rgba(252, 181, 31, 0.18);
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
  }

  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--accent);
    font-size: 0.76rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 10px;
  }

  .eyebrow::before {
    content: "";
    width: 26px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent));
  }

  h1 {
    font-size: clamp(2.1rem, 4vw, 3.25rem);
    line-height: 0.96;
    letter-spacing: -0.045em;
    margin-bottom: 12px;
  }

  .hero-copy {
    max-width: 700px;
    color: var(--text-soft);
    line-height: 1.65;
    font-size: 0.98rem;
  }

  /* ---- Surface (card) ---- */
  .surface {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    overflow: hidden;
    backdrop-filter: blur(16px);
    margin-bottom: 18px;
  }

  .surface-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 18px 20px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.015), transparent);
  }

  .surface-title {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .surface-title h2 {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  .surface-subtitle {
    color: var(--text-muted);
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 4px;
  }

  .surface-icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    background: rgba(252, 181, 31, 0.09);
    color: var(--accent);
    border: 1px solid rgba(252, 181, 31, 0.16);
    flex: 0 0 auto;
  }

  .surface-body {
    padding: 20px;
  }

  /* ---- Agent cards ---- */
  .agent-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
  }

  .agent-card {
    background: rgba(255, 255, 255, 0.025);
    border: 2px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    cursor: pointer;
    transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
  }

  .agent-card:hover {
    border-color: rgba(252, 181, 31, 0.3);
    background: rgba(255, 255, 255, 0.04);
  }

  .agent-card.selected {
    border-color: var(--accent);
    background: rgba(252, 181, 31, 0.06);
    box-shadow: 0 0 0 4px rgba(252, 181, 31, 0.1);
  }

  .agent-card-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    background: rgba(252, 181, 31, 0.09);
    color: var(--accent);
    border: 1px solid rgba(252, 181, 31, 0.16);
    margin-bottom: 12px;
  }

  .agent-card-name {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 6px;
  }

  .agent-card-desc {
    color: var(--text-muted);
    font-size: 0.84rem;
    line-height: 1.5;
  }

  /* ---- Form ---- */
  .form-grid {
    display: grid;
    gap: 18px;
  }

  .form-group {
    display: grid;
    gap: 8px;
  }

  label {
    display: block;
    color: var(--text);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }

  .hint {
    color: var(--text-muted);
    font-size: 0.82rem;
    line-height: 1.5;
  }

  textarea,
  input,
  select,
  button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.025);
    color: var(--text);
    padding: 13px 14px;
    font-size: 0.95rem;
    font-family: inherit;
    transition: border-color 140ms ease, box-shadow 140ms ease, background 140ms ease;
  }

  textarea {
    resize: vertical;
    min-height: 80px;
  }

  textarea::placeholder,
  input::placeholder {
    color: #5d5d66;
  }

  textarea:focus,
  input:focus {
    outline: none;
    border-color: rgba(252, 181, 31, 0.62);
    box-shadow: 0 0 0 4px rgba(252, 181, 31, 0.12);
    background: rgba(255, 255, 255, 0.04);
  }

  button[type="submit"] {
    border: 0;
    background: linear-gradient(180deg, var(--accent-strong), var(--accent));
    color: #111111;
    font-weight: 800;
    cursor: pointer;
    letter-spacing: 0.01em;
  }

  button[type="submit"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 32px rgba(252, 181, 31, 0.18);
  }

  button[type="submit"]:disabled {
    transform: none;
    box-shadow: none;
    background: #55451d;
    color: #948c75;
    cursor: not-allowed;
  }

  /* ---- Execution panel ---- */
  .execution-shell {
    display: grid;
    gap: 14px;
  }

  .execution-summary {
    display: none;
    gap: 12px;
    padding: 16px;
    border-radius: 18px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.025);
  }

  .execution-summary.visible {
    display: grid;
  }

  .execution-summary-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
  }

  .execution-run {
    display: grid;
    gap: 6px;
  }

  .execution-label {
    color: var(--text-muted);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
  }

  .execution-run-name {
    color: var(--text);
    font-size: 0.98rem;
    font-weight: 800;
    word-break: break-word;
  }

  .execution-meta {
    color: var(--text-soft);
    font-size: 0.82rem;
    line-height: 1.5;
  }

  .execution-status {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 112px;
    padding: 9px 12px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid var(--border);
    background: #202020;
    color: #8b8b93;
  }

  .execution-status.running {
    background: rgba(91, 163, 230, 0.16);
    border-color: rgba(91, 163, 230, 0.2);
    color: #7cc1ff;
  }

  .execution-status.succeeded {
    background: rgba(101, 189, 21, 0.18);
    border-color: rgba(101, 189, 21, 0.22);
    color: #9be255;
  }

  .execution-status.failed {
    background: rgba(248, 113, 113, 0.16);
    border-color: rgba(248, 113, 113, 0.2);
    color: var(--danger);
  }

  .execution-links {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }

  .execution-feed {
    display: grid;
    gap: 10px;
  }

  .detail-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: auto;
    text-decoration: none;
    color: #111;
    background: linear-gradient(180deg, var(--accent-strong), var(--accent));
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 0.84rem;
    font-weight: 800;
  }

  .detail-link.secondary {
    color: var(--text);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border);
  }

  .done-banner {
    border-radius: 16px;
    padding: 14px 16px;
    border: 1px solid rgba(101, 189, 21, 0.2);
    background: rgba(101, 189, 21, 0.12);
    color: #9be255;
    font-weight: 700;
    line-height: 1.55;
  }

  .error-banner {
    border-radius: 16px;
    padding: 14px 16px;
    border: 1px solid rgba(248, 113, 113, 0.18);
    background: rgba(248, 113, 113, 0.1);
    color: var(--danger);
    font-weight: 700;
    line-height: 1.55;
  }

  .empty-state {
    border-radius: 16px;
    padding: 14px 16px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.02);
    color: var(--text-muted);
    text-align: center;
    line-height: 1.55;
  }

  /* ---- File upload ---- */
  .file-upload-group {
    display: none;
  }

  .file-upload-group.visible {
    display: grid;
    gap: 8px;
  }

  .file-drop-zone {
    border: 2px dashed rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    cursor: pointer;
    transition: border-color 160ms ease, background 160ms ease;
    background: rgba(255, 255, 255, 0.015);
  }

  .file-drop-zone:hover,
  .file-drop-zone.dragover {
    border-color: rgba(252, 181, 31, 0.5);
    background: rgba(252, 181, 31, 0.04);
  }

  .file-drop-zone.has-file {
    border-color: rgba(101, 189, 21, 0.4);
    background: rgba(101, 189, 21, 0.04);
  }

  .file-drop-icon {
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .file-drop-zone.has-file .file-drop-icon {
    color: #9be255;
  }

  .file-drop-text {
    color: var(--text-muted);
    font-size: 0.88rem;
    line-height: 1.5;
  }

  .file-drop-zone.has-file .file-drop-text {
    color: var(--text-soft);
  }

  .file-name {
    color: var(--text);
    font-weight: 700;
    font-size: 0.9rem;
    margin-top: 4px;
    word-break: break-all;
  }

  .file-hidden-input {
    display: none;
  }

  /* ---- Layout ---- */
  .launch-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
    gap: 18px;
  }

  @media (max-width: 800px) {
    .launch-grid {
      grid-template-columns: 1fr;
    }
    .agent-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
</head>
<body>
<div class="shell">
  <header class="hero">
    <div class="brand-block">
      <div class="brand-mark" aria-hidden="true">
        <svg width="40" height="32" viewBox="0 0 55 44"
          fill="none" xmlns="http://www.w3.org/2000/svg">
          <path fill="#FCB51F" d="M22.24 39.77C14.12
            39.77 7.5 33.51 7.5 25.3V10.99h8.08v19.42
            c0 .9.5 1.4 1.38 1.4h10.55c.88 0 1.38-.5
            1.38-1.4V10.99h8.08V25.3c0 8.21-6.62
            14.47-14.74 14.47Z"/>
          <path fill="#FCB51F" d="M32.76 0c8.11 0
            14.74 6.26 14.74 14.47v14.3h-8.08V9.35
            c0-.89-.5-1.4-1.38-1.4h-10.56c-.88 0-1.38
            .5-1.38 1.4v19.42h-8.08V14.47C18.03 6.26
            24.65 0 32.76 0Z"/>
        </svg>
      </div>
      <div>
        <span class="eyebrow">Union AI Agents</span>
        <h1>Agent Launcher</h1>
        <p class="hero-copy">
          Select an MLE agent, describe your task, and launch it. The agent runs on Union's
          serverless infrastructure and you can track execution status in real time.
        </p>
      </div>
    </div>
  </header>

  <div class="launch-grid">
    <div>
      <div class="surface">
        <div class="surface-header">
          <div class="surface-title">
            <div class="surface-icon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24"
                fill="none" stroke="currentColor" stroke-width="1.8"
                stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/>
                <path d="M12 12l8-4.5"/><path d="M12 12v9"/>
                <path d="M12 12L4 7.5"/>
              </svg>
            </div>
            <div>
              <h2>Launch Agent</h2>
              <p class="surface-subtitle">Choose an agent, enter your prompt, and launch.</p>
            </div>
          </div>
        </div>
        <div class="surface-body">
          <form id="launchForm" class="form-grid">
            <div class="form-group">
              <label>Select Agent</label>
              <div class="agent-grid" id="agentGrid"></div>
            </div>

            <div class="form-group">
              <label for="prompt">Prompt</label>
              <p class="hint">Describe what you want the agent to do with your data.</p>
              <textarea id="prompt" name="prompt" required
                placeholder="e.g. Train a linear regression model to predict target"></textarea>
            </div>

            <div class="file-upload-group" id="fileUploadGroup">
              <label>Data File</label>
              <p class="hint">Upload a CSV or data file for the agent to process.</p>
              <div class="file-drop-zone" id="fileDropZone">
                <div class="file-drop-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24"
                    fill="none" stroke="currentColor" stroke-width="1.5"
                    stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                </div>
                <div class="file-drop-text">Drop a file here or click to browse</div>
                <div class="file-name" id="fileName"></div>
              </div>
              <input type="file" id="fileInput" class="file-hidden-input" accept=".csv,.tsv,.json,.parquet,.txt">
            </div>

            <button type="submit" id="submitBtn" disabled>Select an agent to continue</button>
          </form>
        </div>
      </div>
    </div>

    <div>
      <div class="surface">
        <div class="surface-header">
          <div class="surface-title">
            <div class="surface-icon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24"
                fill="none" stroke="currentColor" stroke-width="1.8"
                stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 7h16"/><path d="M4 12h10"/>
                <path d="M4 17h7"/>
                <circle cx="18" cy="17" r="2.5"/>
              </svg>
            </div>
            <div>
              <h2>Run Activity</h2>
              <p class="surface-subtitle">Execution links and status appear here after launch.</p>
            </div>
          </div>
        </div>
        <div class="surface-body" id="runOutput">
          <div class="empty-state">Select an agent and submit a prompt to see execution status here.</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const agents = AGENT_JSON_PLACEHOLDER;

const agentGrid = document.getElementById('agentGrid');
const form = document.getElementById('launchForm');
const submitBtn = document.getElementById('submitBtn');
const runOutput = document.getElementById('runOutput');
const fileUploadGroup = document.getElementById('fileUploadGroup');
const fileDropZone = document.getElementById('fileDropZone');
const fileInput = document.getElementById('fileInput');
const fileNameEl = document.getElementById('fileName');

let selectedAgent = null;
let selectedFile = null;
let currentRunName = '';
let currentPollToken = 0;
let currentPhase = '';

const SVG_ATTRS = 'width="20" height="20" viewBox="0 0 24 24" ' +
  'fill="none" stroke="currentColor" stroke-width="1.8" ' +
  'stroke-linecap="round" stroke-linejoin="round"';
const ICONS = {
  hammer: `<svg ${SVG_ATTRS}>` +
    '<path d="M15 12l-8.5 8.5a2.12 2.12 0 1 1-3-3L12 9"/>' +
    '<path d="M17.64 15L22 10.64"/>' +
    '<path d="M20.91 11.7l-1.25-1.25a2.83 2.83 0 0 0-4 0' +
    'L14 12.13"/>' +
    '<path d="M18.71 7.04l-3.75-3.75A2.83 2.83 0 0 0 11' +
    ' 3.3L9.34 4.96"/></svg>',
  workflow: `<svg ${SVG_ATTRS}>` +
    '<rect x="3" y="3" width="6" height="6" rx="1"/>' +
    '<rect x="15" y="3" width="6" height="6" rx="1"/>' +
    '<rect x="9" y="15" width="6" height="6" rx="1"/>' +
    '<path d="M6 9v3a1 1 0 0 0 1 1h4"/>' +
    '<path d="M18 9v3a1 1 0 0 1-1 1h-4"/></svg>',
};

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = value ?? '';
  return div.innerHTML;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Build agent selection cards
agents.forEach(agent => {
  const card = document.createElement('div');
  card.className = 'agent-card';
  card.dataset.id = agent.id;
  card.innerHTML = `
    <div class="agent-card-icon">${ICONS[agent.icon] || ICONS.hammer}</div>
    <div class="agent-card-name">${escapeHtml(agent.name)}</div>
    <div class="agent-card-desc">${escapeHtml(agent.description)}</div>
  `;
  card.addEventListener('click', () => {
    document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    selectedAgent = agent;
    submitBtn.disabled = false;
    submitBtn.textContent = `Launch ${agent.name}`;
    // Show/hide file upload based on agent requirements
    if (agent.requires_data) {
      fileUploadGroup.classList.add('visible');
    } else {
      fileUploadGroup.classList.remove('visible');
      selectedFile = null;
      fileInput.value = '';
      fileNameEl.textContent = '';
      fileDropZone.classList.remove('has-file');
    }
  });
  agentGrid.appendChild(card);
});

// File upload interactions
function handleFileSelect(file) {
  if (!file) return;
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileDropZone.classList.add('has-file');
}

fileDropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => handleFileSelect(fileInput.files[0]));

fileDropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  fileDropZone.classList.add('dragover');
});
fileDropZone.addEventListener('dragleave', () => {
  fileDropZone.classList.remove('dragover');
});
fileDropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  fileDropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
});

function statusClass(s) {
  if (/SUCCEEDED/.test(s)) return 'execution-status succeeded';
  if (/FAILED|ABORTED|TIMED_OUT/.test(s)) return 'execution-status failed';
  if (/RUNNING/.test(s)) return 'execution-status running';
  return 'execution-status';
}

function phaseLabel(s) {
  const m = String(s || 'QUEUED').match(/([A-Z_]+)$/);
  return (m ? m[1] : s).replaceAll('_', ' ');
}

function resetRunOutput(agentName) {
  currentPollToken += 1;
  currentRunName = '';
  currentPhase = '';
  runOutput.innerHTML = `
    <div class="execution-shell">
      <div id="execSummary" class="execution-summary visible">
        <div class="execution-summary-top">
          <div class="execution-run">
            <div class="execution-label">Agent</div>
            <div class="execution-run-name">${escapeHtml(agentName)}</div>
            <div class="execution-meta" id="execMeta">Submitting...</div>
          </div>
          <div id="execStatus" class="execution-status">Queued</div>
        </div>
        <div class="execution-links" id="execLinks"></div>
      </div>
      <div id="execFeed" class="execution-feed"></div>
    </div>
  `;
}

function setExecStatus(cls, label, meta) {
  const el = document.getElementById('execStatus');
  const metaEl = document.getElementById('execMeta');
  if (el) { el.className = cls; el.textContent = label; }
  if (metaEl && meta) metaEl.textContent = meta;
}

function setExecRun(runName, runUrl) {
  currentRunName = runName;
  const nameEl = runOutput.querySelector('.execution-run-name');
  const metaEl = document.getElementById('execMeta');
  const linksEl = document.getElementById('execLinks');
  if (nameEl) nameEl.textContent = runName;
  if (metaEl) metaEl.textContent = 'Execution submitted. Polling for status.';
  if (linksEl) {
    const u = escapeHtml(runUrl);
    linksEl.innerHTML =
      `<a class="detail-link" href="${u}"` +
      ` target="_blank" rel="noopener noreferrer"` +
      `>Open Execution</a>` +
      `<a class="detail-link secondary" href="${u}?tab=logs"` +
      ` target="_blank" rel="noopener noreferrer"` +
      `>Open Logs</a>`;
  }
}

function appendMessage(kind, msg) {
  const feed = document.getElementById('execFeed');
  if (!feed) return;
  const el = document.createElement('div');
  el.className = kind;
  el.textContent = msg;
  feed.appendChild(el);
}

async function monitorRun(runName, pollToken) {
  while (pollToken === currentPollToken && currentRunName === runName) {
    try {
      const resp = await fetch(`/api/run-status?run_name=${encodeURIComponent(runName)}`);
      if (!resp.ok) { await sleep(2000); continue; }
      const body = await resp.json();

      if (body.run_url) setExecRun(runName, body.run_url);

      const phase = String(body.phase || '');
      if (phase && phase !== currentPhase) {
        currentPhase = phase;
        if (/SUCCEEDED/.test(phase)) {
          setExecStatus('execution-status succeeded', 'Succeeded', 'Agent completed successfully.');
        } else if (/FAILED|ABORTED|TIMED_OUT/.test(phase)) {
          setExecStatus('execution-status failed', 'Failed', `Execution ended in ${phaseLabel(phase)}.`);
        } else if (/RUNNING/.test(phase)) {
          setExecStatus('execution-status running', phaseLabel(phase), `Agent is ${phaseLabel(phase).toLowerCase()}.`);
        } else {
          setExecStatus('execution-status', phaseLabel(phase), `Agent is ${phaseLabel(phase).toLowerCase()}.`);
        }
      }

      if (body.error_message) {
        appendMessage('error-banner', body.error_message);
      }

      if (body.done) {
        if (body.success) {
          appendMessage('done-banner', 'Agent run completed successfully.');
        }
        return;
      }
    } catch (err) {
      console.error('Poll error:', err);
    }
    await sleep(2000);
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedAgent) return;

  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;

  if (selectedAgent.requires_data && !selectedFile) {
    alert('This agent requires a data file. Please upload one.');
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = 'Launching...';
  resetRunOutput(selectedAgent.name);

  try {
    const formData = new FormData();
    formData.append('agent_id', selectedAgent.id);
    formData.append('prompt', prompt);
    if (selectedFile) {
      formData.append('data_file', selectedFile);
    }

    const resp = await fetch('/api/launch', {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Launch failed');
    }

    const data = await resp.json();
    setExecRun(data.run_name, data.url);
    setExecStatus('execution-status running', 'Running', 'Execution submitted. Polling for status.');

    const token = currentPollToken;
    monitorRun(data.run_name, token);
  } catch (err) {
    appendMessage('error-banner', `Launch failed: ${err.message}`);
    setExecStatus('execution-status failed', 'Failed', 'Could not submit agent run.');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = `Launch ${selectedAgent.name}`;
  }
});
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

import json


@asynccontextmanager
async def lifespan(a: FastAPI):
    await flyte.init_passthrough.aio(
        project=flyte.current_project(),
        domain=flyte.current_domain(),
    )
    yield


app = FastAPI(title="Agent Launcher", lifespan=lifespan)
app.add_middleware(FastAPIPassthroughAuthMiddleware, excluded_paths={"/"})


@app.get("/", response_class=HTMLResponse)
async def index():
    # Inject agent metadata into the HTML so JS can render agent cards
    page = HTML_PAGE.replace(
        "AGENT_JSON_PLACEHOLDER",
        json.dumps(AGENTS),
    )
    return HTMLResponse(page)


@app.post("/api/launch")
async def launch_agent(
    agent_id: str = Form(...),
    prompt: str = Form(...),
    data_file: UploadFile | None = None,
):
    prompt = prompt.strip()
    agent = next((a for a in AGENTS if a["id"] == agent_id), None)
    if not agent:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown agent")
    if not prompt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt is required")
    if agent.get("requires_data") and not data_file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This agent requires a data file")

    try:
        task = remote.Task.get(name=agent["task_name"], auto_version="latest")
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {agent['task_name']} not found: {err}",
        ) from err

    # Build task kwargs
    task_kwargs: dict = {"prompt": prompt}

    # If a data file was uploaded, write to temp, upload to remote storage, and pass as File
    if data_file:
        try:
            suffix = Path(data_file.filename).suffix if data_file.filename else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await data_file.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)

            _md5, remote_uri = await remote.upload_file.aio(tmp_path, fname=data_file.filename)
            task_kwargs["data"] = File.from_existing_remote(remote_uri)
        except Exception as err:
            LOG.exception("Failed to upload data file")
            raise HTTPException(
                status_code=status.HTTP_424_FAILED_DEPENDENCY,
                detail=f"Failed to upload data file: {err}",
            ) from err
        finally:
            tmp_path.unlink(missing_ok=True)

    try:
        run = await flyte.run.aio(task, **task_kwargs)
    except Exception as err:
        LOG.exception("Failed to launch agent run")
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail=f"Failed to launch agent: {err}",
        ) from err

    return JSONResponse(
        {
            "run_name": run.name,
            "url": run.url,
            "agent_id": agent_id,
        }
    )


@app.get("/api/run-status")
async def run_status(run_name: str):
    try:
        run = await remote.Run.get.aio(name=run_name)
        details = await run.details.aio()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail=f"Failed to load status for {run_name}: {err}",
        ) from err

    phase_name = details.action_details.phase.name
    error_message = details.action_details.error_info.message if details.action_details.error_info else ""
    success = phase_name == "SUCCEEDED"

    return JSONResponse(
        {
            "run_name": run_name,
            "run_url": run.url,
            "phase": phase_name,
            "done": details.done(),
            "success": success,
            "error_message": error_message,
        }
    )


@app.get("/api/agents")
async def list_agents():
    return JSONResponse(AGENTS)


# ---------------------------------------------------------------------------
# Flyte app environment
# ---------------------------------------------------------------------------

app_env = FastAPIAppEnvironment(
    name="agent-launcher",
    app=app,
    description="Web console for launching MLE agents with live status tracking.",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "python-multipart"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    scaling=flyte.app.Scaling(replicas=(1, 2)),
    requires_auth=True,
    domain=flyte.app.Domain(subdomain="agent-launcher"),
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    flyte.init_from_config()
    served = flyte.serve(app_env)
    print(f"Agent Launcher served at: {served.url}")
    print(f"OpenAPI docs: {served.endpoint}/docs")
    served.activate()
    input("Press Enter to stop...\n")
