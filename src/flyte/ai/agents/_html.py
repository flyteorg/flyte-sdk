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
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/languages/python.min.js"></script>
    <style>
$CSS
    </style>
</head>
<body>

<div class="sidebar" id="sidebar">
    <div class="sidebar-header">
        <h2>Available Tools</h2>
        <button class="sidebar-toggle" id="sidebarToggle" title="Collapse sidebar">&#x2630;</button>
    </div>
    <div id="toolCards"><p style="color:#666;font-size:13px;">Loading...</p></div>
</div>

<div class="main">
    <button class="sidebar-toggle-float" id="sidebarToggleFloat" title="Expand sidebar">&#x2630;</button>
    <div class="header">$LOGO<h1>$TITLE</h1></div>
    $SUBTITLE
    <div class="nudges" id="nudges"></div>
    <div class="messages" id="messages"></div>
    <div class="clear-chat-bar" id="clearChatBar" style="display:none;">
        <button id="clearChatBtn" title="Clear conversation">Clear chat</button>
    </div>
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

if (typeof marked !== 'undefined' && typeof hljs !== 'undefined') {
    const renderer = new marked.Renderer();
    renderer.code = function({ text, lang }) {
        const language = lang && hljs.getLanguage(lang) ? lang : null;
        const highlighted = language
            ? hljs.highlight(text, { language }).value
            : hljs.highlightAuto(text).value;
        return '<pre><code class="hljs">' + highlighted + '</code></pre>';
    };
    marked.use({ renderer });
}

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
    'Understanding question...',
    'Creating plan...',
    'Looking up information...',
    'Executing plan...',
    'Analyzing results...',
    'Refining response...',
    'Formatting answer...',
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
    }, 4000);

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
    updateClearButton();
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

    html += '<div class="bubble-footer">';
    let metaParts = [];
    if (data.elapsed_ms) metaParts.push('<span>' + (data.elapsed_ms / 1000).toFixed(1) + 's</span>');
    if (data.attempts > 1) metaParts.push('<span>' + data.attempts + ' attempts</span>');
    if (metaParts.length) html += '<div class="meta-badge">' + metaParts.join('') + '</div>';
    if (data.summary) {
        html += '<div class="download-btn-group">'
              + '<button class="download-card-btn" data-fmt="md" title="Download as Markdown">&#x2B07; Download markdown</button>'
              + '<button class="download-card-btn" data-fmt="html" title="Download as HTML">&#x2B07; Download html</button>'
              + '</div>';
    }
    html += '</div>';

    html += '</div>';
    msg.innerHTML = html;
    messagesDiv.appendChild(msg);

    if (data.summary) {
        const summaryText = data.summary;
        msg.querySelector('.download-card-btn[data-fmt="md"]').addEventListener('click', () => downloadFile(summaryText, 'text/markdown', '.md'));
        msg.querySelector('.download-card-btn[data-fmt="html"]').addEventListener('click', () => {
            const rendered = msg.querySelector('.summary-text').innerHTML;
            downloadHtml(rendered);
        });
    }

    executeScripts(msg);
    if (typeof hljs !== 'undefined') {
        msg.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    }
    scrollBottom();
}

function downloadFile(content, mimeType, ext) {
    const blob = new Blob([content + String.fromCharCode(10)], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'response-' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + ext;
    a.click();
    URL.revokeObjectURL(url);
}

function downloadHtml(bodyHtml) {
    const page = '<!DOCTYPE html>'
        + '<html lang="en"><head><meta charset="UTF-8">'
        + '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        + '<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/styles/github.min.css">'
        + '<style>'
        + 'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;'
        + ' max-width: 800px; margin: 40px auto; padding: 0 24px; line-height: 1.7; color: #1a1a2e; background: #fff; }'
        + 'h1,h2,h3,h4 { margin-top: 1.4em; margin-bottom: 0.6em; color: #111; }'
        + 'h2 { border-bottom: 1px solid #e5e7eb; padding-bottom: 0.3em; }'
        + 'p { margin: 0.8em 0; }'
        + 'a { color: #2563eb; text-decoration: none; } a:hover { text-decoration: underline; }'
        + 'ul,ol { padding-left: 1.6em; }'
        + 'li { margin: 0.3em 0; }'
        + 'pre { background: #f6f8fa; border: 1px solid #e5e7eb; border-radius: 8px;'
        + ' padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.5; }'
        + 'pre code.hljs { background: transparent; padding: 0; }'
        + 'code { font-family: "Fira Code", "SF Mono", Consolas, "Liberation Mono", monospace;'
        + ' background: #f0f1f3; padding: 2px 5px; border-radius: 4px; font-size: 0.9em; }'
        + 'pre code { background: none; padding: 0; font-size: inherit; }'
        + 'blockquote { border-left: 3px solid #d1d5db; margin: 1em 0; padding: 0.5em 1em; color: #4b5563; background: #f9fafb; border-radius: 0 6px 6px 0; }'
        + 'strong { color: #111; }'
        + 'hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.5em 0; }'
        + 'table { border-collapse: collapse; width: 100%; margin: 1em 0; }'
        + 'th, td { border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }'
        + 'th { background: #f6f8fa; font-weight: 600; }'
        + '</style></head><body>'
        + bodyHtml
        + '</body></html>';
    downloadFile(page, 'text/html', '.html');
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

function updateClearButton() {
    document.getElementById('clearChatBar').style.display =
        messagesDiv.children.length ? 'flex' : 'none';
}

document.getElementById('clearChatBtn').addEventListener('click', () => {
    messagesDiv.innerHTML = '';
    history = [];
    nudgesDiv.style.display = '';
    updateClearButton();
});

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
    subtitle: str | None = None,
) -> str:
    """Build the full chat HTML with the given *title* and optional *custom_css*.

    The *custom_css* string is injected **after** the default styles, so it
    can override any default rule.  *logo_url*, when provided, renders an
    ``<img>`` to the left of the title in the header bar.

    *subtitle*, when provided, renders a subtitle paragraph below the
    header bar.

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
    desc_html = ""
    if subtitle:
        desc_html = '<p class="app-description">' + subtitle + "</p>"
    action_html = _build_action_buttons_html(additional_buttons or [])
    return (
        CHAT_HTML_TEMPLATE.replace("$TITLE", title)
        .replace("$CSS", css_block)
        .replace("$LOGO", logo_html)
        .replace("$SUBTITLE", desc_html)
        .replace("$ACTION_BUTTONS", action_html)
    )
