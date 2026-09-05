# ruff: noqa: E501
"""Embedded chat UI served at ``GET /``.

A self-contained HTML/CSS/JS string: collapsible sidebar (Tools / Datasets),
a header that shows the LLM model in use, streamed per-phase progress with
timing badges, Chart.js charts, HTML tables and collapsible generated code.
Kept in a ``.py`` file so ``flyte serve`` copies it into the deployment
image alongside the other modules.
"""

CHAT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Analytics Agent</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            display: flex;
            height: 100vh;
            background: #1a1408;
            color: #e0e0e0;
        }

        /* ---------------- Sidebar ---------------- */
        #sidebar {
            width: 300px;
            background: #231c0e;
            border-right: 1px solid rgba(230, 152, 18, 0.2);
            display: flex;
            flex-direction: column;
            transition: width 0.25s ease, opacity 0.25s ease;
            overflow: hidden;
        }
        #sidebar.collapsed { width: 0; border-right: none; }

        #sidebar-collapse-btn {
            position: fixed;
            top: 12px;
            left: 12px;
            z-index: 100;
            width: 32px;
            height: 32px;
            background: rgba(230, 152, 18, 0.15);
            border: 1px solid rgba(230, 152, 18, 0.3);
            border-radius: 8px;
            color: #e69812;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }
        #sidebar-collapse-btn:hover { background: rgba(230, 152, 18, 0.25); }

        #sidebar-toggle {
            display: flex;
            border-bottom: 1px solid rgba(230, 152, 18, 0.2);
            background: #1f170a;
            padding-left: 44px;
        }
        .toggle-button {
            flex: 1;
            padding: 12px;
            background: transparent;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: #999;
            transition: all 0.2s;
        }
        .toggle-button:hover { background: rgba(230, 152, 18, 0.08); }
        .toggle-button.active {
            background: #231c0e;
            color: #e69812;
            border-bottom: 2px solid #e69812;
        }

        #sidebar-content { flex: 1; overflow-y: auto; padding: 16px 14px; }
        .sidebar-panel { display: none; }
        .sidebar-panel.active { display: block; }
        #sidebar h2 {
            margin-bottom: 12px;
            color: #e69812;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .tool-card {
            background: rgba(230, 152, 18, 0.04);
            border: 1px solid rgba(230, 152, 18, 0.12);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tool-card:hover { background: rgba(230, 152, 18, 0.10); border-color: rgba(230, 152, 18, 0.3); }
        .tool-card.expanded { background: rgba(230, 152, 18, 0.08); border-color: rgba(230, 152, 18, 0.3); }
        .tool-header { display: flex; align-items: center; gap: 8px; }
        .tool-expand-icon { color: #e69812; font-size: 10px; transition: transform 0.2s; flex-shrink: 0; }
        .tool-card.expanded .tool-expand-icon { transform: rotate(90deg); }
        .tool-name { font-weight: 600; color: #f2bd52; font-size: 13px; flex-shrink: 0; }
        .tool-brief {
            font-size: 11px;
            color: #888;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .tool-details {
            display: none;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(230, 152, 18, 0.1);
        }
        .tool-card.expanded .tool-details { display: block; }
        .tool-signature {
            font-family: "Fira Code", "SF Mono", Monaco, Consolas, monospace;
            font-size: 11px;
            color: #fad282;
            background: rgba(0, 0, 0, 0.25);
            padding: 6px 8px;
            border-radius: 4px;
            margin-bottom: 8px;
            overflow-x: auto;
            word-break: break-all;
        }
        .tool-description { font-size: 12px; color: #aaa; line-height: 1.5; white-space: pre-wrap; }

        .dataset-item {
            padding: 8px 12px;
            margin-bottom: 6px;
            background: rgba(230, 152, 18, 0.04);
            border: 1px solid rgba(230, 152, 18, 0.12);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .dataset-item:hover { background: rgba(230, 152, 18, 0.12); border-color: #e69812; }
        .dataset-name {
            font-family: "Fira Code", "SF Mono", Monaco, Consolas, monospace;
            font-weight: 600;
            color: #f2bd52;
            font-size: 13px;
            margin-bottom: 2px;
        }
        .dataset-meta { font-size: 11px; color: #888; margin-bottom: 4px; }
        .dataset-columns { font-size: 11px; color: #aaa; line-height: 1.5; }
        .dataset-columns code {
            font-family: "Fira Code", "SF Mono", Monaco, Consolas, monospace;
            background: rgba(0, 0, 0, 0.25);
            padding: 1px 5px;
            border-radius: 4px;
            margin-right: 3px;
        }

        /* ---------------- Main ---------------- */
        #main { flex: 1; display: flex; flex-direction: column; background: #1a1408; min-width: 0; }

        #header {
            padding: 18px 20px 16px 56px;
            border-bottom: 1px solid rgba(230, 152, 18, 0.15);
            background: #1f170a;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }
        #header h1 {
            font-size: 24px;
            background: linear-gradient(90deg, #e69812, #f2bd52);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }
        #header p { color: #999; font-size: 14px; }
        #header-badges { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 8px;
            background: rgba(230, 152, 18, 0.10);
            border: 1px solid rgba(230, 152, 18, 0.25);
            font-size: 12px;
            color: #fad282;
            white-space: nowrap;
        }
        .badge .badge-label { color: #999; text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px; }
        .badge .badge-value { font-family: "Fira Code", "SF Mono", Monaco, Consolas, monospace; }
        .badge .dot { width: 7px; height: 7px; border-radius: 50%; background: #4caf50; box-shadow: 0 0 6px #4caf50; }

        #messages { flex: 1; overflow-y: auto; padding: 24px; }

        .welcome {
            max-width: 720px;
            margin: 40px auto;
            text-align: center;
            color: #aaa;
        }
        .welcome h2 { color: #f2bd52; font-size: 18px; margin-bottom: 8px; }
        .welcome p { font-size: 14px; line-height: 1.6; margin-bottom: 18px; }
        .suggestions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
        .suggestion {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(230, 152, 18, 0.08);
            border: 1px solid rgba(230, 152, 18, 0.25);
            color: #fad282;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .suggestion:hover { background: rgba(230, 152, 18, 0.18); border-color: #e69812; }

        .message { margin-bottom: 20px; max-width: 85%; animation: fadeIn 0.2s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
        .message.user { margin-left: auto; text-align: right; }
        .message.user .message-bubble {
            display: inline-block;
            background: rgba(230, 152, 18, 0.15);
            border: 1px solid rgba(230, 152, 18, 0.3);
            border-radius: 14px 14px 4px 14px;
            padding: 12px 16px;
            text-align: left;
            color: #e0e0e0;
        }
        .message.assistant .message-bubble {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px 14px 14px 4px;
            padding: 16px;
        }
        .message-role {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .message.user .message-role { color: #e69812; }
        .message.assistant .message-role { color: #999; }
        .message-content { line-height: 1.6; white-space: pre-wrap; }

        .chart-container { margin: 14px 0; background: rgba(0, 0, 0, 0.2); padding: 12px; border-radius: 10px; }

        .table-container {
            margin: 14px 0;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(230, 152, 18, 0.15);
            border-radius: 10px;
            overflow: hidden;
        }
        .table-title {
            padding: 10px 14px;
            font-weight: 600;
            font-size: 14px;
            color: #f2bd52;
            border-bottom: 1px solid rgba(230, 152, 18, 0.12);
        }
        .table-scroll { overflow-x: auto; }
        .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .data-table th {
            text-align: left;
            padding: 8px 14px;
            color: #e69812;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: rgba(230, 152, 18, 0.06);
            border-bottom: 1px solid rgba(230, 152, 18, 0.15);
            white-space: nowrap;
        }
        .data-table td { padding: 7px 14px; color: #ddd; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }
        .data-table tbody tr:hover { background: rgba(230, 152, 18, 0.06); }
        .data-table tbody tr:last-child td { border-bottom: none; }

        details { margin-top: 12px; background: rgba(0, 0, 0, 0.3); border-radius: 8px; padding: 10px 14px; }
        summary { cursor: pointer; font-weight: 600; color: #f2bd52; font-size: 13px; }
        summary:hover { color: #fad282; }
        pre {
            margin-top: 8px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.5;
            color: #fad282;
            font-family: "Fira Code", "SF Mono", Monaco, Consolas, monospace;
        }

        .error {
            background: rgba(220, 53, 53, 0.12);
            color: #ff8888;
            padding: 12px 14px;
            border-radius: 0 8px 8px 0;
            border-left: 3px solid #dc3535;
            margin-top: 12px;
            white-space: pre-wrap;
            font-size: 13px;
        }

        #input-area { padding: 16px 24px; border-top: 1px solid rgba(230, 152, 18, 0.15); background: #1f170a; }
        #input-form { display: flex; gap: 12px; }
        #message-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid rgba(230, 152, 18, 0.25);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.05);
            color: #e0e0e0;
            font-size: 14px;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }
        #message-input::placeholder { color: #666; }
        #message-input:focus { border-color: #e69812; }
        #send-button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #e69812, #b8770a);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        #send-button:hover { opacity: 0.9; }
        #send-button:disabled { opacity: 0.5; cursor: not-allowed; }

        .progress {
            margin: 12px 0;
            padding: 12px 14px;
            background: rgba(230, 152, 18, 0.08);
            border-left: 3px solid #e69812;
            border-radius: 0 8px 8px 0;
            font-size: 13px;
            color: #f2bd52;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .progress-text { flex: 1; }
        .progress-text::after { content: '...'; animation: dots 1.5s infinite; }
        @keyframes dots { 0%, 20% { content: '.'; } 40% { content: '..'; } 60%, 100% { content: '...'; } }
        .progress .phase-done { color: #4caf50; font-size: 12px; white-space: nowrap; }
        .progress .phase-retry { color: #ff8888; font-size: 12px; white-space: nowrap; }

        .timing-bar { display: flex; gap: 12px; margin-top: 10px; font-size: 11px; flex-wrap: wrap; }
        .timing-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 8px;
            border-radius: 6px;
            background: rgba(230, 152, 18, 0.10);
            border: 1px solid rgba(230, 152, 18, 0.2);
            color: #fad282;
        }
        .timing-badge .timing-label { color: #999; }
    </style>
</head>
<body>
    <button id="sidebar-collapse-btn" title="Toggle sidebar">&#9776;</button>
    <div id="sidebar">
        <div id="sidebar-toggle">
            <button class="toggle-button active" data-panel="tools">Tools</button>
            <button class="toggle-button" data-panel="datasets">Datasets</button>
        </div>
        <div id="sidebar-content">
            <div class="sidebar-panel active" id="tools-panel">
                <h2>Available Tools</h2>
                <div id="tools-list"></div>
            </div>
            <div class="sidebar-panel" id="datasets-panel">
                <h2>Demo Datasets</h2>
                <div id="datasets-list"></div>
            </div>
        </div>
    </div>

    <div id="main">
        <div id="header">
            <div>
                <h1>Chat Analytics Agent</h1>
                <p>Ask questions in plain English; the agent writes Python, runs it in a Monty sandbox, and returns charts, tables and a summary.</p>
            </div>
            <div id="header-badges">
                <span class="badge"><span class="dot"></span><span class="badge-label">Model</span><span class="badge-value" id="model-badge">loading…</span></span>
                <span class="badge"><span class="badge-label">Sandbox</span><span class="badge-value" id="sandbox-badge">Monty</span></span>
            </div>
        </div>

        <div id="messages">
            <div class="welcome" id="welcome">
                <h2>What would you like to analyze?</h2>
                <p>Four demo datasets are loaded (see the Datasets tab). Try one of these to get started:</p>
                <div class="suggestions" id="suggestions"></div>
            </div>
        </div>

        <div id="input-area">
            <form id="input-form">
                <input type="text" id="message-input" placeholder="Ask a data analysis question..." autocomplete="off" />
                <button type="submit" id="send-button">Send</button>
            </form>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const inputForm = document.getElementById('input-form');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const toolsList = document.getElementById('tools-list');
        const datasetsList = document.getElementById('datasets-list');
        const welcome = document.getElementById('welcome');

        const SUGGESTIONS = [
            'What datasets are available?',
            'Show 2024 revenue by region as a line chart',
            'Rank departments by average salary in a table',
            'Which pages have the highest bounce rate?',
            'Total inventory value by category as a pie chart',
        ];

        let history = [];

        // Sidebar collapse/expand
        const sidebar = document.getElementById('sidebar');
        const collapseBtn = document.getElementById('sidebar-collapse-btn');
        collapseBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            collapseBtn.textContent = sidebar.classList.contains('collapsed') ? '\\u25B6' : '\\u2630';
        });

        // Sidebar tab toggle
        document.querySelectorAll('.toggle-button').forEach(button => {
            button.addEventListener('click', () => {
                const panel = button.dataset.panel;
                document.querySelectorAll('.toggle-button').forEach(b => b.classList.remove('active'));
                button.classList.add('active');
                document.querySelectorAll('.sidebar-panel').forEach(p => p.classList.remove('active'));
                document.getElementById(panel + '-panel').classList.add('active');
            });
        });

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text == null ? '' : String(text);
            return div.innerHTML;
        }

        // First sentence of a description for the collapsed tool card
        function firstLine(desc) {
            const dot = desc.indexOf('.');
            if (dot > 0 && dot < 80) return desc.substring(0, dot + 1);
            if (desc.length > 60) return desc.substring(0, 60) + '...';
            return desc;
        }

        async function loadConfig() {
            try {
                const cfg = await (await fetch('/api/config')).json();
                document.getElementById('model-badge').textContent = cfg.model;
                if (cfg.sandbox) document.getElementById('sandbox-badge').textContent = cfg.sandbox;
            } catch (error) {
                document.getElementById('model-badge').textContent = 'unknown';
            }
        }

        async function loadTools() {
            try {
                const tools = await (await fetch('/api/tools')).json();
                toolsList.innerHTML = tools.map(tool => `
                    <div class="tool-card" onclick="this.classList.toggle('expanded')">
                        <div class="tool-header">
                            <span class="tool-expand-icon">&#9654;</span>
                            <span class="tool-name">${escapeHtml(tool.name)}</span>
                            <span class="tool-brief">${escapeHtml(firstLine(tool.description))}</span>
                        </div>
                        <div class="tool-details">
                            <div class="tool-signature">${escapeHtml(tool.signature)}</div>
                            <div class="tool-description">${escapeHtml(tool.description)}</div>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load tools:', error);
            }
        }

        async function loadDatasets() {
            try {
                const datasets = await (await fetch('/api/datasets')).json();
                datasetsList.innerHTML = datasets.map(ds => `
                    <div class="dataset-item" onclick="insertText('${escapeHtml(ds.name)}')" title="Insert into the prompt">
                        <div class="dataset-name">${escapeHtml(ds.name)}</div>
                        <div class="dataset-meta">${ds.rows} rows &middot; ${ds.columns.length} columns</div>
                        <div class="dataset-columns">${ds.columns.map(c => '<code>' + escapeHtml(c) + '</code>').join('')}</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load datasets:', error);
            }
        }

        function insertText(text) {
            const current = messageInput.value;
            messageInput.value = current && !current.endsWith(' ') ? current + ' ' + text : current + text;
            messageInput.focus();
        }

        function renderSuggestions() {
            const el = document.getElementById('suggestions');
            el.innerHTML = SUGGESTIONS.map(s => `<span class="suggestion">${escapeHtml(s)}</span>`).join('');
            el.querySelectorAll('.suggestion').forEach(chip => {
                chip.addEventListener('click', () => {
                    messageInput.value = chip.textContent;
                    inputForm.requestSubmit();
                });
            });
        }

        function addMessage(role, content, code = null, charts = [], error = null, timing = null) {
            if (welcome) welcome.style.display = 'none';
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            const timestamp = Date.now() + '-' + Math.floor(Math.random() * 1e6);

            let html = '<div class="message-bubble">';
            html += `<div class="message-role">${role}</div>`;
            if (content) html += `<div class="message-content">${escapeHtml(content)}</div>`;

            if (charts && charts.length > 0) {
                charts.forEach((chart, idx) => {
                    html += `<div class="chart-container" id="chart-${timestamp}-${idx}">${chart}</div>`;
                });
            }
            if (code) {
                html += `<details><summary>View generated code</summary><pre>${escapeHtml(code)}</pre></details>`;
            }
            if (error) {
                html += `<div class="error">${escapeHtml(error)}</div>`;
            }
            if (timing) {
                html += '<div class="timing-bar">';
                if (timing.llm_duration_s != null) {
                    html += `<span class="timing-badge"><span class="timing-label">LLM</span>${timing.llm_duration_s}s</span>`;
                }
                if (timing.execution_duration_s != null) {
                    html += `<span class="timing-badge"><span class="timing-label">Sandbox</span>${timing.execution_duration_s}s</span>`;
                }
                if (timing.attempts != null && timing.attempts > 1) {
                    html += `<span class="timing-badge"><span class="timing-label">Attempts</span>${timing.attempts}</span>`;
                }
                html += '</div>';
            }
            html += '</div>';

            messageDiv.innerHTML = html;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Re-run the <script> blocks emitted by create_chart so Chart.js renders
            if (charts && charts.length > 0) {
                setTimeout(() => {
                    charts.forEach((chart, idx) => {
                        const container = document.getElementById(`chart-${timestamp}-${idx}`);
                        if (!container) return;
                        container.querySelectorAll('script').forEach(old => {
                            const s = document.createElement('script');
                            s.textContent = old.textContent;
                            old.parentNode.replaceChild(s, old);
                        });
                    });
                }, 50);
            }
        }

        function createProgress() {
            if (welcome) welcome.style.display = 'none';
            const wrapper = document.createElement('div');
            wrapper.className = 'message assistant';
            const prog = document.createElement('div');
            prog.className = 'progress';
            prog.innerHTML = '<span class="progress-text">Generating code</span>';
            wrapper.appendChild(prog);
            messagesDiv.appendChild(wrapper);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return { wrapper, prog };
        }

        function updateProgress(prog, text, doneText, cls = 'phase-done') {
            if (doneText) {
                const done = document.createElement('span');
                done.className = cls;
                done.textContent = doneText;
                prog.insertBefore(done, prog.querySelector('.progress-text'));
            }
            const active = prog.querySelector('.progress-text');
            if (active) active.textContent = text;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        inputForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const message = messageInput.value.trim();
            if (!message) return;

            addMessage('user', message);
            history.push({ role: 'user', content: message });
            messageInput.value = '';
            sendButton.disabled = true;

            const { wrapper: progressWrapper, prog: progressEl } = createProgress();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, history })
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let finalResult = null;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });

                    const lines = buffer.split('\\n');
                    buffer = lines.pop();  // keep the incomplete line

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const event = JSON.parse(line.slice(6));
                        if (event.phase === 'llm_start') {
                            const label = event.attempt > 1 ? `Generating code (retry ${event.attempt})` : 'Generating code';
                            updateProgress(progressEl, label, null);
                        } else if (event.phase === 'llm_done') {
                            updateProgress(progressEl, 'Running in sandbox', `LLM ${event.llm_duration_s}s \\u2713`);
                        } else if (event.phase === 'retry') {
                            updateProgress(progressEl, `Retrying (attempt ${event.attempt + 1})`, 'sandbox error \\u21bb', 'phase-retry');
                        } else if (event.phase === 'done') {
                            finalResult = event;
                        }
                    }
                }

                progressWrapper.remove();

                if (finalResult) {
                    addMessage(
                        'assistant',
                        finalResult.summary || (finalResult.error ? '' : 'Analysis complete'),
                        finalResult.code,
                        finalResult.charts,
                        finalResult.error,
                        {
                            llm_duration_s: finalResult.llm_duration_s,
                            execution_duration_s: finalResult.execution_duration_s,
                            attempts: finalResult.attempts,
                        }
                    );
                    const assistantText = finalResult.summary || finalResult.error;
                    if (assistantText) history.push({ role: 'assistant', content: assistantText });
                } else {
                    addMessage('assistant', '', null, [], 'The server closed the stream without a result.');
                }
            } catch (error) {
                progressWrapper.remove();
                addMessage('assistant', '', null, [], `Error: ${error.message}`);
            } finally {
                sendButton.disabled = false;
                messageInput.focus();
            }
        });

        loadConfig();
        loadTools();
        loadDatasets();
        renderSuggestions();
        messageInput.focus();
    </script>
</body>
</html>
"""
