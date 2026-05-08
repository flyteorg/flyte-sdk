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
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ccircle cx='8' cy='8' r='8' fill='%236F2AEF'/%3E%3C/svg%3E" />
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
        <h2>Available tools <span class="tool-count-wrap" id="toolCountWrap" hidden><span id="toolCount" class="tool-count-badge">0</span></span></h2>
        <button class="sidebar-toggle" id="sidebarToggle" title="Collapse sidebar">&#x2630;</button>
    </div>
    <div class="tool-sidebar-toolbar" id="toolSidebarToolbar" hidden>
        <input type="search" id="toolFilter" class="tool-filter-input" placeholder="Filter tools…" autocomplete="off" />
        <div class="tool-toolbar-row">
            <button type="button" class="tool-toolbar-btn" id="expandAllTools">Expand all</button>
            <span class="tool-toolbar-sep">·</span>
            <button type="button" class="tool-toolbar-btn" id="collapseAllTools">Collapse all</button>
        </div>
    </div>
    <div class="tool-cards-scroll" id="toolCardsScroll">
        <div id="toolCards"><p class="tool-cards-loading">Loading…</p></div>
    </div>
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
let allTools = [];

try {
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
} catch (e) {
    console.warn('Chat UI: markdown/highlighter setup failed', e);
}

document.getElementById('sidebarToggle').addEventListener('click', () => {
    sidebar.classList.add('collapsed');
});
document.getElementById('sidebarToggleFloat').addEventListener('click', () => {
    sidebar.classList.remove('collapsed');
});

function truncateText(s, maxLen) {
    if (!s || s.length <= maxLen) return s || '';
    return s.slice(0, maxLen - 1).trim() + '\u2026';
}

function buildToolCard(t) {
    const card = document.createElement('div');
    card.className = 'tool-card';
    const displayName = (t.name || '').replace(/_/g, ' ');
    const desc = t.description || '';
    const preview = truncateText(desc.replace(/\\s+/g, ' '), 72);

    const header = document.createElement('div');
    header.className = 'tool-card-header';
    header.setAttribute('role', 'button');
    header.setAttribute('aria-expanded', 'false');
    header.innerHTML =
        '<div class="tool-card-header-text">'
        + '<h3>' + escapeHtml(displayName) + '</h3>'
        + '<p class="tool-card-preview">' + escapeHtml(preview) + '</p>'
        + '</div>'
        + '<span class="tool-card-chevron" aria-hidden="true">&#x25B6;</span>';
    header.addEventListener('click', () => {
        const on = card.classList.toggle('expanded');
        header.setAttribute('aria-expanded', on ? 'true' : 'false');
    });

    const body = document.createElement('div');
    body.className = 'tool-card-body';
    body.innerHTML =
        '<div class="tool-card-body-inner">'
        + '<p class="tool-card-desc">' + escapeHtml(desc) + '</p>'
        + '<div class="sig">' + escapeHtml(t.signature || '') + '</div>'
        + '</div>';

    card.appendChild(header);
    card.appendChild(body);
    return card;
}

function renderToolCards(tools) {
    const container = document.getElementById('toolCards');
    const countEl = document.getElementById('toolCount');
    const countWrap = document.getElementById('toolCountWrap');
    const toolbar = document.getElementById('toolSidebarToolbar');
    container.innerHTML = '';
    countEl.textContent = tools.length === allTools.length
        ? String(tools.length)
        : tools.length + ' / ' + allTools.length;
    countWrap.hidden = false;
    toolbar.hidden = allTools.length === 0;
    if (allTools.length === 0) {
        countWrap.hidden = true;
        container.innerHTML = '<p class="tool-cards-empty">No tools registered for this agent.</p>';
        return;
    }
    if (tools.length === 0) {
        container.innerHTML = '<p class="tool-cards-empty">No tools match your filter.</p>';
        return;
    }
    tools.forEach(t => container.appendChild(buildToolCard(t)));
}

(async () => {
    const toolCardsEl = document.getElementById('toolCards');
    try {
        const resp = await fetch('/api/tools');
        if (!resp.ok) {
            toolCardsEl.innerHTML =
                '<p class="tool-cards-error">Failed to load tools (HTTP ' + resp.status + ')</p>';
            return;
        }
        const tools = await resp.json();
        allTools = Array.isArray(tools) ? tools : [];
        renderToolCards(allTools);
    } catch (e) {
        toolCardsEl.innerHTML =
            '<p class="tool-cards-error">Failed to load tools</p>';
    }
})();

const toolFilterEl = document.getElementById('toolFilter');
if (toolFilterEl) {
    toolFilterEl.addEventListener('input', e => {
        const q = (e.target.value || '').trim().toLowerCase();
        if (!q) {
            renderToolCards(allTools);
            return;
        }
        const filtered = allTools.filter(t => {
            const name = (t.name || '').toLowerCase();
            const desc = (t.description || '').toLowerCase();
            const sig = (t.signature || '').toLowerCase();
            return name.includes(q) || desc.includes(q) || sig.includes(q);
        });
        renderToolCards(filtered);
    });
}

const expandAllToolsBtn = document.getElementById('expandAllTools');
if (expandAllToolsBtn) {
    expandAllToolsBtn.addEventListener('click', () => {
        document.querySelectorAll('#toolCards .tool-card').forEach(card => {
            card.classList.add('expanded');
            const h = card.querySelector('.tool-card-header');
            if (h) h.setAttribute('aria-expanded', 'true');
        });
    });
}
const collapseAllToolsBtn = document.getElementById('collapseAllTools');
if (collapseAllToolsBtn) {
    collapseAllToolsBtn.addEventListener('click', () => {
        document.querySelectorAll('#toolCards .tool-card').forEach(card => {
            card.classList.remove('expanded');
            const h = card.querySelector('.tool-card-header');
            if (h) h.setAttribute('aria-expanded', 'false');
        });
    });
}

(async () => {
    try {
        const resp = await fetch('/api/nudges');
        if (!resp.ok) {
            nudgesDiv.style.display = 'none';
            return;
        }
        const raw = await resp.json();
        const nudges = Array.isArray(raw) ? raw : [];
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
    } catch (e) {
        nudgesDiv.style.display = 'none';
    }
})();

// Step 0 is client-only until the first server progress event; steps 1-3 align with
// CodeModeAgent phases (generating_code / executing / formatting).
const PROGRESS_STEP_LABELS = [
    'Preparing runtime environment...',
    'Creating plan...',
    'Executing plan...',
    'Formatting answer...',
];

const CODE_MODE_PHASE_TO_STEP = {
    generating_code: 1,
    executing: 2,
    formatting: 3,
};

function createPendingAssistantBubble() {
    const msg = document.createElement('div');
    msg.className = 'msg assistant assistant-pending';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    const panel = document.createElement('div');
    panel.className = 'generating-panel';
    const label = document.createElement('div');
    label.className = 'generating-label';
    label.textContent = 'Working on your answer';
    const sub = document.createElement('div');
    sub.className = 'generating-sub';
    sub.textContent = 'In progress…';
    const track = document.createElement('div');
    track.className = 'progress-steps';
    PROGRESS_STEP_LABELS.forEach((text, i) => {
        const step = document.createElement('div');
        step.className = 'progress-step pending';
        step.dataset.stepIndex = String(i);
        const dot = document.createElement('span');
        dot.className = 'progress-step-dot';
        const tx = document.createElement('span');
        tx.className = 'progress-step-text';
        tx.textContent = text;
        step.appendChild(dot);
        step.appendChild(tx);
        track.appendChild(step);
    });
    panel.appendChild(label);
    panel.appendChild(sub);
    panel.appendChild(track);
    bubble.appendChild(panel);
    msg.appendChild(bubble);
    return { msg, track, sub };
}

function applyCodemodeProgress(trackEl, subEl, evt) {
    if (!evt || !evt.phase) return;
    const idx = CODE_MODE_PHASE_TO_STEP[evt.phase];
    if (idx !== undefined) setProgressUI(trackEl, idx);
    if (subEl && evt.attempt != null && evt.max_attempts != null) {
        subEl.textContent = 'Attempt ' + evt.attempt + ' of ' + evt.max_attempts;
    }
}

function setProgressUI(trackEl, phaseIndex) {
    if (!trackEl) return;
    trackEl.querySelectorAll('.progress-step').forEach((el, i) => {
        el.classList.remove('done', 'active', 'pending');
        if (phaseIndex < 0) {
            el.classList.add('pending');
        } else if (i < phaseIndex) el.classList.add('done');
        else if (i === phaseIndex) el.classList.add('active');
        else el.classList.add('pending');
    });
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    nudgesDiv.style.display = 'none';
    appendUser(text);
    userInput.value = '';
    sendBtn.disabled = true;

    const { msg: pendingMsg, track: progressTrack, sub: progressSub } = createPendingAssistantBubble();
    messagesDiv.appendChild(pendingMsg);
    updateClearButton();
    scrollBottom();

    // Step 0 until server sends generating_code (CodeModeAgent, in-process run only).
    setProgressUI(progressTrack, 0);

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: text, history: history, stream: true }),
        });

        if (!resp.ok) {
            let errMsg = 'Server error (' + resp.status + ')';
            try {
                const body = await resp.json();
                if (body.detail) errMsg += ': ' + JSON.stringify(body.detail);
            } catch(_) {}
            if (pendingMsg.parentNode) pendingMsg.remove();
            appendAssistant({ error: errMsg });
        } else {
            const ct = (resp.headers.get('Content-Type') || '').toLowerCase();
            let data = null;
            if (ct.includes('ndjson') && resp.body) {
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop() || '';
                    for (const line of lines) {
                        if (!line.trim()) continue;
                        let obj;
                        try { obj = JSON.parse(line); } catch (_) { continue; }
                        if (obj.type === 'progress') {
                            applyCodemodeProgress(progressTrack, progressSub, obj);
                            scrollBottom();
                        } else if (obj.type === 'done') {
                            data = obj;
                        }
                    }
                }
                if (buffer.trim()) {
                    try {
                        const obj = JSON.parse(buffer);
                        if (obj.type === 'done') data = obj;
                    } catch (_) {}
                }
            } else {
                data = await resp.json();
            }

            if (!data) {
                if (pendingMsg.parentNode) pendingMsg.remove();
                appendAssistant({ error: 'Empty response from server' });
            } else {
                if (!pendingMsg.parentNode) {
                    appendAssistant(data);
                } else {
                    pendingMsg.classList.remove('assistant-pending');
                    const bubble = pendingMsg.querySelector('.bubble');
                    bubble.innerHTML = '';
                    fillAssistantBubble(bubble, data);
                    wireAssistantBubble(pendingMsg, data);
                }
                history.push({ role: 'user', content: text });
                const assistantContent = data.summary || data.error || '';
                if (assistantContent) {
                    history.push({ role: 'assistant', content: assistantContent });
                }
            }
        }
    } catch(e) {
        if (pendingMsg.parentNode) pendingMsg.remove();
        appendAssistant({ error: 'Request failed: ' + e.message });
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
        scrollBottom();
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

function fillAssistantBubble(bubble, data) {
    let html = '';

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
              + '<pre><code class="language-python">' + escapeHtml(data.code) + '</code></pre></details>';
    }

    html += '<div class="bubble-footer">';
    const metaParts = [];
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

    bubble.innerHTML = html;
}

function wireAssistantBubble(msg, data) {
    if (data.summary) {
        const summaryText = data.summary;
        const mdBtn = msg.querySelector('.download-card-btn[data-fmt="md"]');
        const htmlBtn = msg.querySelector('.download-card-btn[data-fmt="html"]');
        if (mdBtn) mdBtn.addEventListener('click', () => downloadFile(summaryText, 'text/markdown', '.md'));
        if (htmlBtn) htmlBtn.addEventListener('click', () => {
            const st = msg.querySelector('.summary-text');
            if (st) downloadHtml(st.innerHTML);
        });
    }
    executeScripts(msg);
    if (typeof hljs !== 'undefined') {
        msg.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    }
    scrollBottom();
}

function appendAssistant(data) {
    const msg = document.createElement('div');
    msg.className = 'msg assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    fillAssistantBubble(bubble, data);
    msg.appendChild(bubble);
    messagesDiv.appendChild(msg);
    wireAssistantBubble(msg, data);
    updateClearButton();
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
