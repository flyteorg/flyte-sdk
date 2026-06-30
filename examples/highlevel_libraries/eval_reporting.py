"""
Evaluation reporting module for generating Plotly visualizations.

This module provides functions to generate interactive HTML reports
for LLM evaluation results.
"""

from typing import Dict, List, Optional

import flyte.report


async def generate_evaluation_report(
    eval_results: List[Dict[str, str]], aggregated: Optional[Dict[str, str]], num_samples: int
) -> None:
    """Generate a Plotly visualization report for evaluation results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Extract all unique metric keys from results
    all_keys = set()
    for result in eval_results:
        all_keys.update(result.keys())

    # Filter out non-numeric metrics for visualization
    numeric_metrics = {}
    for key in all_keys:
        try:
            # Try to convert all values to float
            values = [float(result.get(key, "0")) for result in eval_results]
            numeric_metrics[key] = values
        except (ValueError, TypeError):
            # Skip non-numeric metrics
            pass

    # Determine number of subplots needed
    num_metrics = len(numeric_metrics)
    if num_metrics == 0:
        # No numeric metrics, create a simple summary table
        await generate_text_report(eval_results, aggregated, num_samples)
        return

    # Create subplots - distribution plots for each metric
    num_cols = min(2, num_metrics)
    num_rows = (num_metrics + num_cols - 1) // num_cols

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[f"{key.replace('_', ' ').title()} Distribution" for key in numeric_metrics.keys()],
        specs=[[{"type": "box"}] * num_cols for _ in range(num_rows)],
    )

    # Add box plots for each metric
    for idx, (metric_name, values) in enumerate(numeric_metrics.items()):
        row = idx // num_cols + 1
        col = idx % num_cols + 1

        fig.add_trace(
            go.Box(
                y=values,
                name=metric_name,
                marker_color="rgb(102, 126, 234)",
                boxmean="sd",  # Show mean and standard deviation
            ),
            row=row,
            col=col,
        )

        # Update y-axis title
        fig.update_yaxes(title_text="Value", row=row, col=col)

    # Update layout
    fig.update_layout(
        title_text=f"Evaluation Results Dashboard ({num_samples} samples)",
        showlegend=False,
        height=300 * num_rows,
        template="plotly_white",
    )

    # Create aggregated metrics summary if available
    agg_html = ""
    if aggregated:
        agg_html = "<div style='margin: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px;'>"
        agg_html += "<h2 style='color: #667eea; margin-bottom: 15px;'>Aggregated Metrics</h2>"
        agg_html += (
            "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>"
        )

        for key, value in aggregated.items():
            agg_html += f"""
            <div style='background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.9em; color: #666; text-transform: uppercase; margin-bottom: 5px;'>
                    {key.replace("_", " ")}
                </div>
                <div style='font-size: 1.8em; font-weight: bold; color: #667eea;'>
                    {value}
                </div>
            </div>
            """

        agg_html += "</div></div>"

    # Create per-sample results table
    table_html = "<div style='margin: 20px;'>"
    table_html += "<h2 style='color: #667eea; margin-bottom: 15px;'>Individual Sample Results</h2>"
    table_html += "<div style='overflow-x: auto;'>"
    table_html += "<table style='width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>"  # noqa: E501

    # Table header
    if eval_results:
        headers = list(eval_results[0].keys())
        table_html += "<thead><tr style='background: #667eea; color: white;'>"
        table_html += "<th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Sample #</th>"
        for header in headers:
            table_html += f"<th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>{header.replace('_', ' ').title()}</th>"  # noqa: E501
        table_html += "</tr></thead>"

        # Table body
        table_html += "<tbody>"
        for idx, result in enumerate(eval_results):
            row_style = "background: #f8f9fa;" if idx % 2 == 0 else "background: white;"
            table_html += f"<tr style='{row_style}'>"
            table_html += f"<td style='padding: 10px; border: 1px solid #ddd; font-weight: bold;'>{idx + 1}</td>"
            for header in headers:
                value = result.get(header, "N/A")
                table_html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{value}</td>"
            table_html += "</tr>"
        table_html += "</tbody>"

    table_html += "</table></div></div>"

    # Combine everything into final HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluation Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{
                color: #667eea;
                text-align: center;
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 LLM Evaluation Report</h1>
            {agg_html}
            {fig.to_html(full_html=False, include_plotlyjs="cdn")}
            {table_html}
        </div>
    </body>
    </html>
    """

    await flyte.report.replace.aio(html_content)
    await flyte.report.flush.aio()


async def generate_text_report(
    eval_results: List[Dict[str, str]], aggregated: Optional[Dict[str, str]], num_samples: int
) -> None:
    """Generate a simple text report when no numeric metrics are available."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Evaluation Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
            }}
            h1 {{ color: #667eea; }}
            .metric {{ padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Evaluation Report</h1>
            <p><strong>Total Samples:</strong> {num_samples}</p>
            <h2>Results</h2>
    """

    for idx, result in enumerate(eval_results):
        html_content += f"<div class='metric'><strong>Sample {idx + 1}:</strong> {result}</div>"

    if aggregated:
        html_content += "<h2>Aggregated Metrics</h2>"
        for key, value in aggregated.items():
            html_content += f"<div class='metric'><strong>{key}:</strong> {value}</div>"

    html_content += "</div></body></html>"

    await flyte.report.replace.aio(html_content)
    await flyte.report.flush.aio()
