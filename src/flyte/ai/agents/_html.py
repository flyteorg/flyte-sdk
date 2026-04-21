"""HTML template and builder for the Agent Chat UI."""

from __future__ import annotations

from ._css import DEFAULT_CSS

CHAT_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$TITLE</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
$CSS
    </style>
</head>
<body>

<div class="sidebar" id="sidebar">
    <div class="sidebar-header">
        <h2>Available Tools</h2>
        <button class="sidebar-toggle" id="sidebarToggle" title="Collapse sidebar">&#x25C0;</button>
    </div>
    <div id="toolCards"><p style="color:#666;font-size:13px;">Loading...</p></div>
</div>

<div class="main">
    <button class="sidebar-toggle-float" id="sidebarToggleFloat" title="Expand sidebar">&#x25B6;</button>
    <div class="header">$LOGO<h1>$TITLE</h1></div>
    <div class="nudges" id="nudges"></div>
    <div class="messages" id="messages"></div>
    <div class="input-bar">
        <input type="text" id="userInput"
               placeholder="Ask a question..."
               autocomplete="off" />
        <button id="sendBtn" onclick="sendMessage()">Send</button>
        $ACTION_BUTTONS
    </div>
</div>

<script>
const messagesDiv = document.getElementById('messages');
const userInput   = document.getElementById('userInput');
const sendBtn     = document.getElementById('sendBtn');
const nudgesDiv   = document.getElementById('nudges');
const sidebar     = document.getElementById('sidebar');

let history = [];

document.getElementById('sidebarToggle').addEventListener('click', () => {
    sidebar.classList.add('collapsed');
});
document.getElementById('sidebarToggleFloat').addEventListener('click', () => {
    sidebar.classList.remove('collapsed');
});

(async () => {
    try {
        const resp = await fetch('/api/tools');
        const tools = await resp.json();
        const container = document.getElementById('toolCards');
        container.innerHTML = '';
        tools.forEach(t => {
            const card = document.createElement('div');
            card.className = 'tool-card';

            const header = document.createElement('div');
            header.className = 'tool-card-header';
            header.innerHTML =
                '<h3>' + escapeHtml(t.name.replaceAll('_', ' ')) + '</h3>'
                + '<span class="tool-card-chevron">&#x25B6;</span>';
            header.addEventListener('click', () => card.classList.toggle('expanded'));

            const body = document.createElement('div');
            body.className = 'tool-card-body';
            body.innerHTML =
                '<div class="tool-card-body-inner">'
                + '<p>' + escapeHtml(t.description) + '</p>'
                + '<div class="sig">' + escapeHtml(t.signature) + '</div>'
                + '</div>';

            card.appendChild(header);
            card.appendChild(body);
            container.appendChild(card);
        });
    } catch(e) {
        document.getElementById('toolCards').innerHTML =
            '<p style="color:#ff8888;font-size:13px;">Failed to load tools</p>';
    }
})();

(async () => {
    try {
        const resp = await fetch('/api/nudges');
        const nudges = await resp.json();
        nudgesDiv.innerHTML = '';
        nudges.forEach(n => {
            const card = document.createElement('div');
            card.className = 'nudge-card';
            card.innerHTML = '<h4>' + escapeHtml(n.label) + '</h4>'
                           + '<p>' + escapeHtml(n.prompt) + '</p>';
            card.addEventListener('click', () => {
                userInput.value = n.prompt;
                nudgesDiv.style.display = 'none';
                sendMessage();
            });
            nudgesDiv.appendChild(card);
        });
        if (!nudges.length) nudgesDiv.style.display = 'none';
    } catch(e) {
        nudgesDiv.style.display = 'none';
    }
})();

const PROGRESS_PHASES = [
    'Creating plan...',
    'Executing plan...',
    'Refining results...',
];

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    nudgesDiv.style.display = 'none';
    appendUser(text);
    userInput.value = '';
    sendBtn.disabled = true;

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

        clearInterval(progressTimer);
        statusEl.remove();

        if (!resp.ok) {
            let errMsg = 'Server error (' + resp.status + ')';
            try {
                const body = await resp.json();
                if (body.detail) errMsg += ': ' + JSON.stringify(body.detail);
            } catch(_) {}
            appendAssistant({ error: errMsg });
        } else {
            const data = await resp.json();
            appendAssistant(data);
            history.push({ role: 'user', content: text });
            const assistantContent = data.summary || data.error || '';
            if (assistantContent) {
                history.push({ role: 'assistant', content: assistantContent });
            }
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

    if (data.summary) {
        html += '<div class="summary-text">'
              + (typeof marked !== 'undefined' ? marked.parse(data.summary) : escapeHtml(data.summary))
              + '</div>';
    }

    if (data.charts && data.charts.length) {
        data.charts.forEach(chartHtml => {
            html += '<div class="chart-container">' + chartHtml + '</div>';
        });
    }

    if (data.error) {
        html += '<div class="error-box">' + escapeHtml(data.error) + '</div>';
    }

    if (data.code) {
        html += '<details><summary>Executed Plan</summary>'
              + '<pre>' + escapeHtml(data.code) + '</pre></details>';
    }

    let metaParts = [];
    if (data.elapsed_ms) metaParts.push('<span>' + (data.elapsed_ms / 1000).toFixed(1) + 's</span>');
    if (data.attempts > 1) metaParts.push('<span>' + data.attempts + ' attempts</span>');
    if (metaParts.length) html += '<div class="meta-badge">' + metaParts.join('') + '</div>';

    html += '</div>';
    msg.innerHTML = html;
    messagesDiv.appendChild(msg);

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

userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !sendBtn.disabled) {
        e.preventDefault();
        sendMessage();
    }
});

(function() {
    const chevron = document.getElementById('actionChevron');
    const dropup = document.getElementById('actionDropup');
    if (!chevron || !dropup) return;
    chevron.addEventListener('click', e => {
        e.stopPropagation();
        dropup.classList.toggle('open');
    });
    document.addEventListener('click', () => dropup.classList.remove('open'));
})();
</script>
</body>
</html>
"""


def _build_action_buttons_html(buttons: list[dict[str, str]]) -> str:
    if not buttons:
        return ""
    first = buttons[0]
    has_menu = len(buttons) > 1
    cls = "action-btn-group has-menu" if has_menu else "action-btn-group"
    parts = [
        '<div class="' + cls + '">',
        '  <a class="action-primary" href="'
        + first["button_url"]
        + '" target="_blank" rel="noopener">'
        + first["button_text"]
        + "</a>",
    ]
    if has_menu:
        parts.append('  <button class="action-chevron" id="actionChevron" title="More actions">&#x25B2;</button>')
        parts.append('  <div class="action-dropup" id="actionDropup">')
        for btn in buttons[1:]:
            parts.append(
                '    <a href="' + btn["button_url"] + '" target="_blank" rel="noopener">' + btn["button_text"] + "</a>"
            )
        parts.append("  </div>")
    parts.append("</div>")
    return "\n".join(parts)


def build_chat_html(
    title: str = "Agent Chat",
    custom_css: str = "",
    logo_url: str | None = None,
    additional_buttons: list[dict[str, str]] | None = None,
) -> str:
    """Build the full chat HTML with the given *title* and optional *custom_css*.

    The *custom_css* string is injected **after** the default styles, so it
    can override any default rule.  *logo_url*, when provided, renders an
    ``<img>`` to the left of the title in the header bar.

    *additional_buttons* is an optional list of ``{"button_text": ...,
    "button_url": ...}`` dicts.  The first entry becomes the primary
    (prominent) button; the rest appear in a drop-up menu behind a chevron.
    """
    css_block = DEFAULT_CSS
    if custom_css:
        css_block += "\n/* --- Custom overrides --- */\n" + custom_css
    logo_html = ""
    if logo_url:
        logo_html = '<img class="header-logo" src="' + logo_url + '" alt="logo" />'
    action_html = _build_action_buttons_html(additional_buttons or [])
    return (
        CHAT_HTML_TEMPLATE.replace("$TITLE", title)
        .replace("$CSS", css_block)
        .replace("$LOGO", logo_html)
        .replace("$ACTION_BUTTONS", action_html)
    )
