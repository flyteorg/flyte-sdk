"""
HTML templates for Human-in-the-Loop (HITL) events.

This module contains all HTML template strings used by the HITL plugin.
"""


def get_index_page() -> str:
    """Landing page with instructions."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HITL Event Service</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
            .endpoint { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Human-in-the-Loop Event Service</h1>
        <p>This service allows humans to provide input to paused Flyte workflow events.</p>

        <div class="endpoint">
            <h3>Submit Input</h3>
            <p>To submit input for a pending event, visit:</p>
            <code>GET /form/{request_id}</code>
            <p>Or POST directly to:</p>
            <code>POST /submit</code>
        </div>

        <div class="endpoint">
            <h3>Check Event Status</h3>
            <code>GET /status/{request_id}</code>
        </div>
    </body>
    </html>
    """


def get_bool_select_element() -> str:
    """HTML select element for boolean input."""
    return """
            <select
                name="value"
                required
                style="
                    width: 100%;
                    padding: 12px;
                    font-size: 16px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    box-sizing: border-box;
                    margin-bottom: 15px;"
                onchange="this.form.submit()"
            >
                <option value="">-- Select --</option>
                <option value="true">True</option>
                <option value="false">False</option>
            </select>
        """


def get_input_form_page(
    event_name: str,
    prompt: str,
    data_type_name: str,
    request_id: str,
    response_path: str,
    input_element: str,
) -> str:
    """HTML page for the input form."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Event Input Required</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 500px;
                margin: 25px auto;
                padding: 20px;
            }}
            .card {{
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{ color: #333; margin-top: 0; }}
            .event-name {{ color: #4CAF50; font-weight: bold; margin-bottom: 10px; }}
            .prompt {{ color: #666; margin-bottom: 20px; }}
            .request-id {{ font-size: 12px; color: #999; margin-bottom: 20px; }}
            .data-type {{ font-size: 12px; color: #666; margin-bottom: 10px; }}
            input[type="number"], input[type="text"] {{
                width: 100%;
                padding: 12px;
                font-size: 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
                margin-bottom: 15px;
            }}
            button {{
                width: 100%;
                padding: 12px;
                font-size: 16px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{ background: #45a049; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Event Input Required</h1>
            <p class="event-name">Event: {event_name}</p>
            <p class="prompt">{prompt}</p>
            <p class="data-type">Expected type: <code>{data_type_name}</code></p>
            <p class="request-id">Request ID: {request_id}</p>
            <form action="/submit" method="post">
                <input type="hidden" name="request_id" value="{request_id}">
                <input type="hidden" name="data_type" value="{data_type_name}">
                <input type="hidden" name="response_path" value="{response_path}">
                {input_element}
                <button type="submit">Submit</button>
            </form>
        </div>
    </body>
    </html>
    """


def get_submission_success_page(
    request_id: str,
    value: str,
    data_type: str,
    message: str,
) -> str:
    """HTML page displayed after successful submission."""
    import html as html_module

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Submission Successful</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 500px;
                margin: 25px auto;
                padding: 20px;
            }}
            .card {{
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{ color: #4CAF50; margin-top: 0; }}
            .success-icon {{
                font-size: 48px;
                text-align: center;
                margin-bottom: 20px;
            }}
            .message {{ color: #666; margin-bottom: 20px; font-size: 16px; }}
            .details {{
                background: #f9f9f9;
                border-radius: 6px;
                padding: 15px;
                margin-top: 20px;
            }}
            .detail-row {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            .detail-row:last-child {{ border-bottom: none; }}
            .detail-label {{ font-weight: bold; color: #333; }}
            .detail-value {{ color: #666; font-family: monospace; }}
            .status-badge {{
                display: inline-block;
                background: #4CAF50;
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="success-icon">&#10004;</div>
            <h1>Submission Successful</h1>
            <p class="message">{html_module.escape(message)}</p>
            <div class="details">
                <div class="detail-row">
                    <span class="detail-label">Status</span>
                    <span class="status-badge">Submitted</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Request ID</span>
                    <span class="detail-value">{html_module.escape(request_id)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Value</span>
                    <span class="detail-value">{html_module.escape(str(value))}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Data Type</span>
                    <span class="detail-value">{html_module.escape(data_type)}</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_event_report_html(
    form_url: str,
    api_url: str,
    curl_body: str,
    type_name: str,
) -> str:
    """HTML report for the event input form shown in the Flyte UI."""
    import html as html_module

    return f"""
        <style>
            .hitl-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 20px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                color: white;
            }}
            .hitl-header {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .hitl-header h1 {{
                margin: 0;
                font-size: 1.8em;
            }}
            .hitl-section {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
                color: #333;
            }}
            .hitl-section h2 {{
                margin-top: 0;
                color: #667eea;
                font-size: 1.2em;
                border-bottom: 2px solid #667eea;
                padding-bottom: 8px;
            }}
            .hitl-url {{
                background: #f5f5f5;
                padding: 12px;
                border-radius: 6px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9em;
                word-break: break-all;
                border-left: 4px solid #667eea;
            }}
            .hitl-url a {{
                color: #667eea;
                text-decoration: none;
            }}
            .hitl-url a:hover {{
                text-decoration: underline;
            }}
            .hitl-code {{
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 15px;
                border-radius: 6px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.85em;
                overflow-x: auto;
                white-space: pre-wrap;
                word-break: break-all;
            }}
            .hitl-info {{
                display: grid;
                grid-template-columns: 120px 1fr;
                gap: 8px;
                margin-bottom: 15px;
            }}
            .hitl-info-label {{
                font-weight: bold;
                color: #667eea;
            }}
            .hitl-info-value {{
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9em;
            }}
        </style>
        <div class="hitl-container">
            <div class="hitl-header">
                <h1>Event Input Required</h1>
            </div>

            <div class="hitl-section">
                <h2>Option 1: Web Form</h2>
                <p>
                Submit your input using <a href="{html_module.escape(form_url)}" target="_blank">this form</a>:
                </p>
            </div>

            <div class="hitl-section">
                <h2>Option 2: Programmatic API (curl)</h2>
                <p>Use the following curl command to submit input programmatically:</p>
                <div class="hitl-code">curl -X POST "{html_module.escape(api_url)}" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {{FLYTE_API_KEY}}" \\
  -d '{html_module.escape(curl_body)}'</div>
                <p style="margin-top: 15px; font-size: 0.9em; color: #666;">
                    <strong>Note:</strong> Replace <code>{{your_value}}</code> with the actual value you want to
                    submit. The value should match the expected type: <code>{html_module.escape(type_name)}</code>

                    <p>Replace <code>{{FLYTE_API_KEY}}</code> with your Flyte API key.</p>
                </p>
            </div>
        </div>
        """
