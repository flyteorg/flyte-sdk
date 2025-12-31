"""
Batch Inference Report Generation

Generates interactive HTML reports from batch inference results.
Uses Flyte's reporting capabilities to create beautiful visualizations
with Chart.js showing label distribution, confidence statistics, and detailed results.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa

import flyte
import flyte.io
import flyte.report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the same image as batch inference
batch_image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=Path("pyproject.toml"), extra_args="--extra batch"
)

# Report environment
report_env = flyte.TaskEnvironment(
    name="batch_inference_report",
    image=batch_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)


@report_env.task(report=True)
async def generate_batch_inference_report(results_df: flyte.io.DataFrame):
    """
    Generate an interactive HTML report from batch inference results.

    This task creates a beautiful, interactive dashboard showing:
    - Summary statistics (total images, success rate, avg confidence)
    - Label distribution chart
    - Confidence distribution histogram
    - Detailed results table

    Args:
        results_df: DataFrame with inference results from batch_inference_pipeline
    """
    # Load DataFrame
    table = await results_df.open(pa.Table).all()
    df = table.to_pandas()

    # Calculate statistics
    total_images = len(df)
    successful = df["error"].isna().sum()
    failed = df["error"].notna().sum()
    success_rate = (successful / total_images * 100) if total_images > 0 else 0.0

    # Confidence statistics
    successful_df = df[df["error"].isna()]
    if len(successful_df) > 0:
        avg_confidence = successful_df["top_confidence"].mean() * 100

        # Label distribution
        label_counts = successful_df["top_label"].value_counts().to_dict()
        labels = list(label_counts.keys())
        counts = list(label_counts.values())

        # Confidence distribution (binned)
        confidence_bins = [0, 50, 60, 70, 80, 90, 100]
        confidence_hist, _ = pd.cut(successful_df["top_confidence"] * 100, bins=confidence_bins, retbins=True)
        confidence_counts = confidence_hist.value_counts().sort_index()
        confidence_labels = [f"{int(interval.left)}-{int(interval.right)}%" for interval in confidence_counts.index]
        confidence_values = confidence_counts.tolist()
    else:
        avg_confidence = 0.0
        labels = []
        counts = []
        confidence_labels = []
        confidence_values = []

    # Prepare table data (top 100 results for display)
    table_rows = []
    for idx, row in df.head(100).iterrows():
        if pd.isna(row["error"]):
            table_rows.append(
                {
                    "path": row["image_path"],
                    "label": row["top_label"],
                    "confidence": f"{row['top_confidence'] * 100:.2f}%",
                    "second": f"{row['second_label']} ({row['second_confidence'] * 100:.1f}%)"
                    if pd.notna(row["second_label"])
                    else "-",
                    "status": "✅ Success",
                }
            )
        else:
            table_rows.append(
                {
                    "path": row["image_path"],
                    "label": "-",
                    "confidence": "-",
                    "second": "-",
                    "status": f"❌ Error: {row['error'][:50]}",
                }
            )

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Batch Inference Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }}
            .header h1 {{
                font-size: 2.5em;
                font-weight: 700;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                margin-bottom: 10px;
            }}
            .header p {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                position: relative;
                overflow: hidden;
            }}
            .metric-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }}
            .metric-value {{
                font-size: 2.2em;
                font-weight: 700;
                color: #667eea;
                display: block;
                margin-bottom: 5px;
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .charts {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 25px;
                margin-bottom: 30px;
            }}
            .chart-container {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }}
            .chart-title {{
                font-size: 1.3em;
                font-weight: 600;
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }}
            .chart-wrapper {{
                position: relative;
                height: 300px;
            }}
            .table-container {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }}
            th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                position: sticky;
                top: 0;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #eee;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            .truncate {{
                max-width: 300px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Batch Inference Report</h1>
                <p>Image Classification Results</p>
            </div>

            <div class="metrics">
                <div class="metric-card">
                    <span class="metric-value">{total_images:,}</span>
                    <span class="metric-label">Total Images</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{success_rate:.1f}%</span>
                    <span class="metric-label">Success Rate</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{avg_confidence:.1f}%</span>
                    <span class="metric-label">Avg Confidence</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{successful:,}</span>
                    <span class="metric-label">Successful</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{failed:,}</span>
                    <span class="metric-label">Failed</span>
                </div>
            </div>

            <div class="charts">
                <div class="chart-container">
                    <div class="chart-title">Label Distribution</div>
                    <div class="chart-wrapper">
                        <canvas id="labelChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Confidence Distribution</div>
                    <div class="chart-wrapper">
                        <canvas id="confidenceChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="table-container">
                <div class="chart-title">Detailed Results (Top 100)</div>
                <table>
                    <thead>
                        <tr>
                            <th>Image Path</th>
                            <th>Top Prediction</th>
                            <th>Confidence</th>
                            <th>Second Best</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {
        "".join(
            [
                f'''
                        <tr>
                            <td class="truncate">{row["path"]}</td>
                            <td>{row["label"]}</td>
                            <td>{row["confidence"]}</td>
                            <td>{row["second"]}</td>
                            <td>{row["status"]}</td>
                        </tr>
                        '''
                for row in table_rows
            ]
        )
    }
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            // Label Distribution Chart
            const labelCtx = document.getElementById('labelChart').getContext('2d');
            new Chart(labelCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        label: 'Predictions',
                        data: {json.dumps(counts)},
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: '#667eea',
                        borderWidth: 1,
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});

            // Confidence Distribution Chart
            const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
            new Chart(confidenceCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(confidence_labels)},
                    datasets: [{{
                        label: 'Images',
                        data: {json.dumps(confidence_values)},
                        backgroundColor: 'rgba(118, 75, 162, 0.8)',
                        borderColor: '#764ba2',
                        borderWidth: 1,
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """

    await flyte.report.replace.aio(html_content)
    await flyte.report.flush.aio()

    logger.info("Interactive report generated successfully")


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    print("Batch Inference Report Generation Module - Import this module to use the report generation task")
