"""Default CSS for the Agent Chat UI."""

DEFAULT_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0f;
    color: #d1d5db;
    height: 100vh;
    display: flex;
}

/* --- Left sidebar: tool cards --- */
.sidebar {
    width: 280px;
    min-width: 280px;
    background: #0e0e14;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    display: flex;
    flex-direction: column;
    padding: 20px 16px;
    overflow: hidden;
    min-height: 0;
    transition: width 0.2s ease, min-width 0.2s ease, padding 0.2s ease;
}
.tool-cards-scroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    margin: 0 -4px;
    padding: 0 4px 8px;
    /* Subtle scrollbar (avoids harsh default / light track on dark UI) */
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.18) transparent;
}
.tool-cards-scroll::-webkit-scrollbar {
    width: 6px;
}
.tool-cards-scroll::-webkit-scrollbar-track {
    background: transparent;
}
.tool-cards-scroll::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.14);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
}
.tool-cards-scroll::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.26);
    background-clip: padding-box;
}
.tool-cards-loading,
.tool-cards-error,
.tool-cards-empty {
    font-size: 13px;
    color: #6b7280;
    line-height: 1.45;
    padding: 4px 2px 8px;
}
.tool-cards-error { color: #f87171; }
.tool-count-wrap {
    font-weight: 500;
    letter-spacing: 0;
    text-transform: none;
}
.tool-count-badge {
    display: inline-block;
    margin-left: 6px;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    color: #9ca3af;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.06);
    vertical-align: middle;
}
.tool-sidebar-toolbar {
    margin-bottom: 12px;
    flex-shrink: 0;
}
.tool-filter-input {
    width: 100%;
    padding: 8px 10px;
    margin-bottom: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.03);
    color: #d1d5db;
    font-size: 12px;
    outline: none;
    transition: border-color 0.15s;
}
.tool-filter-input::placeholder { color: #4b5563; }
.tool-toolbar-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 2px 6px;
}
.tool-toolbar-btn {
    background: none;
    border: none;
    color: #6b7280;
    font-size: 11px;
    cursor: pointer;
    padding: 2px 0;
    text-decoration: underline;
    text-underline-offset: 2px;
}
.tool-toolbar-btn:hover { color: #d1d5db; }
.tool-toolbar-sep { color: #4b5563; font-size: 11px; user-select: none; }
.sidebar.collapsed {
    width: 0;
    min-width: 0;
    padding: 0;
    overflow: hidden;
    border-right: none;
}
.sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}
.sidebar-header h2 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6b7280;
    margin: 0;
}
.sidebar-toggle {
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 10;
    width: 30px;
    height: 30px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    color: #9ca3af;
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s, color 0.15s;
    padding: 0;
    line-height: 1;
}
.sidebar-toggle:hover {
    background: rgba(111, 42, 239, 0.12);
    color: #d1d5db;
}
.sidebar:not(.collapsed) .sidebar-toggle {
    position: static;
    flex-shrink: 0;
}
.sidebar.collapsed .sidebar-toggle {
    display: none;
}
.sidebar.collapsed ~ .main .sidebar-toggle-float {
    display: flex;
}
.sidebar-toggle-float {
    display: none;
    position: absolute;
    top: 13px;
    left: 16px;
    z-index: 20;
    width: 30px;
    height: 30px;
    background: rgba(14, 14, 20, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    color: #9ca3af;
    font-size: 16px;
    cursor: pointer;
    align-items: center;
    justify-content: center;
    transition: background 0.15s, color 0.15s;
    padding: 0;
    line-height: 1;
}
.sidebar-toggle-float:hover {
    background: rgba(111, 42, 239, 0.12);
    color: #d1d5db;
}
.tool-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0;
    margin-bottom: 10px;
    overflow: hidden;
}
.tool-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 8px;
    padding: 10px 14px;
    cursor: pointer;
    user-select: none;
    transition: background 0.12s;
}
.tool-card-header-text {
    min-width: 0;
    flex: 1;
}
.tool-card-preview {
    display: block;
    margin-top: 4px;
    font-size: 11px;
    color: #6b7280;
    line-height: 1.35;
}
.tool-card.expanded .tool-card-preview {
    display: none;
}
.tool-card-header .tool-card-chevron {
    margin-top: 2px;
}
.tool-card-header:hover {
    background: rgba(255, 255, 255, 0.03);
}
.tool-card h3 {
    font-size: 13px;
    color: #e5e7eb;
    margin: 0;
}
.tool-card-chevron {
    font-size: 11px;
    color: #6b7280;
    transition: transform 0.2s ease;
    flex-shrink: 0;
    margin-left: 8px;
}
.tool-card.expanded .tool-card-chevron {
    transform: rotate(90deg);
}
.tool-card-body {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.25s ease;
}
.tool-card.expanded .tool-card-body {
    max-height: 400px;
}
.tool-card-body-inner {
    padding: 0 14px 12px;
}
.tool-card p.tool-card-desc {
    font-size: 12px;
    color: #9ca3af;
    line-height: 1.5;
    margin-bottom: 8px;
}
.tool-card .sig {
    font-family: 'Fira Code', Consolas, monospace;
    font-size: 11px;
    color: #9B70EF;
    word-break: break-all;
}

/* --- Main chat area --- */
.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    position: relative;
}

.header {
    padding: 16px 24px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    background: #0c0c12;
    transition: padding-left 0.2s ease;
    display: flex;
    align-items: center;
    gap: 12px;
}
.header:has(+ .app-description) {
    border-bottom: none;
}
.sidebar.collapsed ~ .main .header {
    padding-left: 58px;
}
.header-logo {
    height: 28px;
    width: auto;
    object-fit: contain;
    flex-shrink: 0;
}
.header h1 {
    font-size: 18px;
    font-weight: 600;
    color: #e5e7eb;
}
.app-description {
    padding: 6px 24px 10px;
    font-size: 13px;
    color: #9ca3af;
    line-height: 1.4;
    background: #0c0c12;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.sidebar.collapsed ~ .main .app-description {
    padding-left: 58px;
}

.messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.18) transparent;
}
.messages::-webkit-scrollbar {
    width: 6px;
}
.messages::-webkit-scrollbar-track {
    background: transparent;
}
.messages::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.14);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
}
.messages::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.26);
    background-clip: padding-box;
}

/* --- Message bubbles --- */
.msg {
    max-width: 85%;
    margin-bottom: 20px;
    animation: fadeIn 0.2s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: none; }
}

.msg.user {
    margin-left: auto;
    text-align: right;
}
.msg.user .bubble {
    display: inline-block;
    background: rgba(111, 42, 239, 0.12);
    border: 1px solid rgba(111, 42, 239, 0.22);
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    text-align: left;
    color: #d1d5db;
}

.msg.assistant .bubble {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px 14px 14px 4px;
    padding: 16px;
}

.msg.assistant-pending .bubble {
    border-color: rgba(111, 42, 239, 0.22);
    background: rgba(111, 42, 239, 0.04);
    animation: pendingPulse 2.2s ease-in-out infinite;
}
@keyframes pendingPulse {
    0%, 100% { border-color: rgba(111, 42, 239, 0.18); }
    50% { border-color: rgba(111, 42, 239, 0.38); }
}

.generating-panel {
    padding: 2px 0 4px;
}
.generating-label {
    font-size: 14px;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 4px;
}
.generating-sub {
    font-size: 12px;
    color: #6b7280;
    line-height: 1.4;
    margin-bottom: 16px;
}
.progress-steps {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.progress-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 5px 0;
    transition: opacity 0.2s;
}
.progress-step-dot {
    flex-shrink: 0;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-top: 4px;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.08);
    transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.progress-step-text {
    font-size: 12px;
    line-height: 1.45;
    color: #6b7280;
    transition: color 0.2s;
}
.progress-step.done .progress-step-dot {
    background: rgba(111, 42, 239, 0.85);
    border-color: rgba(111, 42, 239, 0.5);
}
.progress-step.done .progress-step-text {
    color: #9ca3af;
}
.progress-step.active .progress-step-dot {
    background: rgba(111, 42, 239, 0.95);
    border-color: rgba(167, 139, 250, 0.9);
    box-shadow: 0 0 0 3px rgba(111, 42, 239, 0.2);
    animation: stepPulse 1.2s ease-in-out infinite;
}
.progress-step.active .progress-step-text {
    color: #e5e7eb;
    font-weight: 500;
}
.progress-step.pending .progress-step-text {
    color: #4b5563;
}
@keyframes stepPulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.08); opacity: 0.85; }
}

.msg.assistant details {
    margin-top: 12px;
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 10px 14px;
}
.msg.assistant details summary {
    cursor: pointer;
    font-weight: 600;
    color: #9B70EF;
    font-size: 13px;
}
.msg.assistant details pre {
    background: rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    padding: 10px 12px;
}
.msg.assistant details pre code {
    background: none;
    padding: 0;
    font-size: 13px;
}
.msg.assistant details pre code.hljs {
    background: transparent;
    padding: 0;
}
.msg.assistant pre {
    margin-top: 8px;
    overflow-x: auto;
    font-size: 12px;
    line-height: 1.5;
    color: #9ca3af;
    font-family: 'Fira Code', Consolas, monospace;
}

.chart-container {
    margin-top: 14px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 12px;
}

.summary-text {
    margin-top: 12px;
    line-height: 1.6;
    font-size: 14px;
    color: #d1d5db;
}
.summary-text h1, .summary-text h2, .summary-text h3,
.summary-text h4, .summary-text h5, .summary-text h6 {
    color: #e5e7eb;
    margin: 16px 0 8px;
}
.summary-text h1 { font-size: 20px; }
.summary-text h2 { font-size: 17px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 6px; }
.summary-text h3 { font-size: 15px; }
.summary-text p { margin: 8px 0; }
.summary-text ul, .summary-text ol { margin: 8px 0; padding-left: 24px; }
.summary-text li { margin: 4px 0; }
.summary-text a { color: #9B70EF; text-decoration: none; }
.summary-text a:hover { text-decoration: underline; }
.summary-text code {
    background: rgba(255,255,255,0.06);
    padding: 2px 5px;
    border-radius: 4px;
    font-family: 'Fira Code', Consolas, monospace;
    font-size: 13px;
}
.summary-text pre {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 6px;
    padding: 10px 12px;
    overflow-x: auto;
    margin: 10px 0;
}
.summary-text pre code {
    background: none;
    padding: 0;
    font-size: 13px;
}
.summary-text pre code.hljs {
    background: transparent;
    padding: 0;
}
.summary-text strong { color: #e5e7eb; }
.summary-text hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 16px 0;
}
.summary-text blockquote {
    border-left: 2px solid rgba(111,42,239,0.3);
    padding-left: 12px;
    margin: 8px 0;
    color: #9ca3af;
}

.bubble-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 10px;
    gap: 8px;
}
.meta-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: #6b7280;
}
.meta-badge span {
    background: rgba(255,255,255,0.04);
    padding: 2px 8px;
    border-radius: 4px;
}
.download-btn-group {
    display: inline-flex;
    gap: 4px;
    margin-left: auto;
}
.download-card-btn {
    background: none;
    border: none;
    color: #6b7280;
    font-size: 11px;
    cursor: pointer;
    padding: 2px 8px;
    border-radius: 4px;
    transition: color 0.15s, background 0.15s;
}
.download-card-btn:hover {
    color: #d1d5db;
    background: rgba(255, 255, 255, 0.06);
}

.error-box {
    margin-top: 12px;
    background: rgba(220, 53, 53, 0.08);
    border-left: 2px solid #dc3535;
    border-radius: 0 6px 6px 0;
    padding: 12px 14px;
    color: #f87171;
    font-size: 13px;
}

/* --- Input bar --- */
.input-bar {
    padding: 16px 24px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: #0c0c12;
    display: flex;
    gap: 12px;
}
.input-bar input {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.03);
    color: #d1d5db;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
}
.input-bar input:focus {
    border-color: rgba(111, 42, 239, 0.5);
}
.input-bar input::placeholder { color: #4b5563; }
.input-bar button#sendBtn {
    padding: 12px 24px;
    background: rgba(255, 255, 255, 0.06);
    color: #d1d5db;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}
.input-bar button#sendBtn:hover {
    background: rgba(255, 255, 255, 0.10);
    border-color: rgba(255, 255, 255, 0.18);
}
.input-bar button#sendBtn:disabled { opacity: 0.4; cursor: not-allowed; }

/* --- Clear chat button --- */
.clear-chat-bar {
    display: flex;
    justify-content: center;
    padding: 4px 20px 12px;
}
.clear-chat-bar button {
    background: none;
    border: none;
    color: #6b7280;
    font-size: 12px;
    cursor: pointer;
    padding: 4px 12px;
    border-radius: 6px;
    transition: color 0.15s, background 0.15s;
}
.clear-chat-bar button:hover {
    color: #d1d5db;
    background: rgba(255, 255, 255, 0.06);
}

/* --- Additional action button (primary + dropup) --- */
.action-btn-group {
    position: relative;
    display: flex;
    align-items: stretch;
}
.action-btn-group .action-primary {
    padding: 12px 20px;
    background: #6F2AEF;
    color: #f3f4f6;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.15s;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
}
.action-btn-group.has-menu .action-primary {
    border-radius: 8px 0 0 8px;
}
.action-btn-group .action-primary:hover { background: #8B52F2; }
.action-btn-group .action-chevron {
    padding: 0 10px;
    background: #6F2AEF;
    color: #f3f4f6;
    border: none;
    border-left: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 0 8px 8px 0;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    transition: background 0.15s;
}
.action-btn-group .action-chevron:hover { background: #8B52F2; }
.action-dropup {
    display: none;
    position: absolute;
    bottom: calc(100% + 6px);
    right: 0;
    min-width: 180px;
    background: #16161e;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 4px 0;
    box-shadow: 0 -4px 16px rgba(0, 0, 0, 0.4);
    z-index: 30;
}
.action-dropup.open { display: block; }
.action-dropup a {
    display: block;
    padding: 10px 16px;
    color: #d1d5db;
    text-decoration: none;
    font-size: 13px;
    transition: background 0.12s;
}
.action-dropup a:hover { background: rgba(255, 255, 255, 0.06); }

.typing {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #6b7280;
    font-style: italic;
    font-size: 13px;
    margin-bottom: 16px;
}
.typing::before {
    content: '';
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.08);
    border-top-color: #6b7280;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* --- Prompt nudge cards --- */
.nudges {
    flex: 1;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    align-content: center;
    width: 75%;
    max-width: 75%;
    margin: 0 auto;
    padding: 32px 24px 8px;
}
@media (max-width: 1024px) {
    .nudges { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 768px) {
    .nudges { width: 100%; max-width: 100%; grid-template-columns: 1fr; }
}
.nudge-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 14px 18px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s, transform 0.1s;
    display: flex;
    flex-direction: column;
}
.nudge-card:hover {
    background: rgba(111, 42, 239, 0.08);
    border-color: rgba(111, 42, 239, 0.25);
    transform: translateY(-1px);
}
.nudge-card h4 {
    font-size: 13px;
    color: #d1d5db;
    margin-bottom: 4px;
}
.nudge-card p {
    font-size: 12px;
    color: #6b7280;
    line-height: 1.4;
    margin: 0;
}
"""
