"""Embedded chat UI served at ``GET /``.

This is a self-contained HTML/CSS/JS string that renders a Chat.js-powered
analytics interface.  Kept in a ``.py`` file so ``flyte serve`` copies it
into the deployment image alongside the other modules.
"""

CHAT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Analytics Agent</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1408;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
        }

        /* --- Left sidebar: tool cards --- */
        .sidebar {
            width: 280px;
            min-width: 280px;
            background: #231c0e;
            border-right: 1px solid rgba(230, 152, 18, 0.2);
            display: flex;
            flex-direction: column;
            padding: 20px 16px;
            overflow-y: auto;
        }
        .sidebar h2 {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #e69812;
            margin-bottom: 16px;
        }
        .tool-card {
            background: rgba(230, 152, 18, 0.06);
            border: 1px solid rgba(230, 152, 18, 0.15);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .tool-card h3 {
            font-size: 14px;
            color: #f2bd52;
            margin-bottom: 4px;
        }
        .tool-card .sig {
            font-family: 'Fira Code', Consolas, monospace;
            font-size: 11px;
            color: #fad282;
            margin-bottom: 8px;
            word-break: break-all;
        }
        .tool-card p {
            font-size: 12px;
            color: #aaa;
            line-height: 1.5;
        }

        /* --- Main chat area --- */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        .header {
            padding: 16px 24px;
            border-bottom: 1px solid rgba(230, 152, 18, 0.15);
            background: #1f170a;
        }
        .header h1 {
            font-size: 20px;
            background: linear-gradient(90deg, #e69812, #f2bd52);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        /* --- Message bubbles --- */
        .msg {
            max-width: 85%;
            margin-bottom: 20px;
            animation: fadeIn 0.2s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

        .msg.user {
            margin-left: auto;
            text-align: right;
        }
        .msg.user .bubble {
            display: inline-block;
            background: rgba(230, 152, 18, 0.15);
            border: 1px solid rgba(230, 152, 18, 0.3);
            border-radius: 14px 14px 4px 14px;
            padding: 12px 16px;
            text-align: left;
        }

        .msg.assistant .bubble {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px 14px 14px 4px;
            padding: 16px;
        }

        /* Code block inside assistant bubble */
        .msg.assistant details {
            margin-top: 12px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 10px 14px;
        }
        .msg.assistant details summary {
            cursor: pointer;
            font-weight: 600;
            color: #f2bd52;
            font-size: 13px;
        }
        .msg.assistant pre {
            margin-top: 8px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.5;
            color: #fad282;
            font-family: 'Fira Code', Consolas, monospace;
        }

        /* Chart container */
        .chart-container {
            margin-top: 14px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 12px;
        }

        /* Summary text */
        .summary-text {
            margin-top: 12px;
            background: rgba(230, 152, 18, 0.08);
            border-left: 3px solid #e69812;
            border-radius: 0 8px 8px 0;
            padding: 12px 14px;
            line-height: 1.6;
            font-size: 14px;
        }

        /* Error box */
        .error-box {
            margin-top: 12px;
            background: rgba(220, 53, 53, 0.12);
            border-left: 3px solid #dc3535;
            border-radius: 0 8px 8px 0;
            padding: 12px 14px;
            color: #ff8888;
            font-size: 13px;
        }

        /* --- Input bar --- */
        .input-bar {
            padding: 16px 24px;
            border-top: 1px solid rgba(230, 152, 18, 0.15);
            background: #1f170a;
            display: flex;
            gap: 12px;
        }
        .input-bar input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid rgba(230, 152, 18, 0.25);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.05);
            color: #e0e0e0;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        .input-bar input:focus {
            border-color: #e69812;
        }
        .input-bar input::placeholder { color: #666; }
        .input-bar button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #e69812, #b8770a);
            color: #fff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .input-bar button:hover { opacity: 0.9; }
        .input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Spinner */
        .typing { color: #999; font-style: italic; font-size: 13px; margin-bottom: 16px; }
    </style>
</head>
<body>

<div class="sidebar">
    <h2>Available Tools</h2>
    <div id="toolCards"><p style="color:#666;font-size:13px;">Loading...</p></div>
</div>

<div class="main">
    <div class="header">
        <h1>Chat Analytics Agent</h1>
    </div>

    <div class="messages" id="messages"></div>

    <div class="input-bar">
        <input type="text" id="userInput"
               placeholder="Ask a data analysis question..."
               autocomplete="off" />
        <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
const messagesDiv = document.getElementById('messages');
const userInput   = document.getElementById('userInput');
const sendBtn     = document.getElementById('sendBtn');

// Conversation history sent to the server (text only, no chart HTML)
let history = [];

// ---- Load tool cards from /api/tools ----
(async () => {
    try {
        const resp = await fetch('/api/tools');
        const tools = await resp.json();
        const container = document.getElementById('toolCards');
        container.innerHTML = '';
        tools.forEach(t => {
            const card = document.createElement('div');
            card.className = 'tool-card';
            card.innerHTML =
                '<h3>' + escapeHtml(t.name) + '</h3>' +
                '<div class="sig">' + escapeHtml(t.signature) + '</div>' +
                '<p>' + escapeHtml(t.description) + '</p>';
            container.appendChild(card);
        });
    } catch(e) {
        document.getElementById('toolCards').innerHTML =
            '<p style="color:#ff8888;font-size:13px;">Failed to load tools</p>';
    }
})();

// ---- Send message ----
const PROGRESS_PHASES = [
    'Generating analysis code...',
    'Running code in sandbox...',
    'Building charts...',
];

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendUser(text);
    userInput.value = '';
    sendBtn.disabled = true;

    // Animated progress indicator that cycles through phases
    const statusEl = document.createElement('div');
    statusEl.className = 'typing';
    statusEl.textContent = PROGRESS_PHASES[0];
    messagesDiv.appendChild(statusEl);
    scrollBottom();

    let phase = 0;
    const progressTimer = setInterval(() => {
        phase = Math.min(phase + 1, PROGRESS_PHASES.length - 1);
        statusEl.textContent = PROGRESS_PHASES[phase];
    }, 3000);

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: text, history: history }),
        });
        const data = await resp.json();

        clearInterval(progressTimer);
        statusEl.remove();
        appendAssistant(data);

        // Update history with text-only entries
        history.push({ role: 'user', content: text });
        const assistantContent = data.summary || data.error || '';
        if (assistantContent) {
            history.push({ role: 'assistant', content: assistantContent });
        }
    } catch(e) {
        clearInterval(progressTimer);
        statusEl.remove();
        appendAssistant({ error: 'Request failed: ' + e.message });
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// ---- Render helpers ----
function appendUser(text) {
    const msg = document.createElement('div');
    msg.className = 'msg user';
    msg.innerHTML = '<div class="bubble">' + escapeHtml(text) + '</div>';
    messagesDiv.appendChild(msg);
    scrollBottom();
}

function appendAssistant(data) {
    const msg = document.createElement('div');
    msg.className = 'msg assistant';

    let html = '<div class="bubble">';

    // Collapsible code
    if (data.code) {
        html += '<details><summary>Generated Code</summary>'
              + '<pre>' + escapeHtml(data.code) + '</pre></details>';
    }

    // Charts
    if (data.charts && data.charts.length) {
        data.charts.forEach(chartHtml => {
            html += '<div class="chart-container">' + chartHtml + '</div>';
        });
    }

    // Summary
    if (data.summary) {
        html += '<div class="summary-text">' + escapeHtml(data.summary) + '</div>';
    }

    // Error
    if (data.error) {
        html += '<div class="error-box">' + escapeHtml(data.error) + '</div>';
    }

    html += '</div>';
    msg.innerHTML = html;
    messagesDiv.appendChild(msg);

    // Re-execute <script> tags so Chart.js renders
    executeScripts(msg);
    scrollBottom();
}

function executeScripts(container) {
    container.querySelectorAll('script').forEach(old => {
        const s = document.createElement('script');
        s.textContent = old.textContent;
        old.parentNode.replaceChild(s, old);
    });
}

function scrollBottom() {
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

// Enter key sends message
userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !sendBtn.disabled) {
        e.preventDefault();
        sendMessage();
    }
});
</script>
</body>
</html>
"""
