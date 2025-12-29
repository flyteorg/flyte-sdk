"""
Batch OCR Comparison Report Generation

Generates interactive HTML reports comparing multiple OCR models.
Shows success rates, text quality metrics, processing statistics,
and side-by-side comparisons with beautiful visualizations using Chart.js.
"""

import json
import logging

import flyte
import flyte.io
import flyte.report
import pandas as pd
import pyarrow as pa

from batch_ocr import driver_env, ocr_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Report environment
report_env = flyte.TaskEnvironment(
    name="ocr_comparison_report",
    image=ocr_image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    depends_on=[driver_env],
)


@report_env.task(report=True)
async def generate_ocr_comparison_report(results_dfs: list[flyte.io.DataFrame]):
    """
    Generate an interactive HTML report comparing multiple OCR models.

    This creates a comprehensive dashboard with:
    - Model comparison matrix (success rates, avg tokens, etc.)
    - Success rate comparison chart
    - Token count distribution per model
    - Error analysis
    - Side-by-side text comparisons for sample documents

    Args:
        results_dfs: List of DataFrames, one per model
    """
    logger.info(f"Generating OCR comparison report for {len(results_dfs)} models")

    # Load all DataFrames
    all_dfs = []
    for df in results_dfs:
        table = await df.open(pa.Table).all()
        pandas_df = table.to_pandas()
        all_dfs.append(pandas_df)

    # Combine all results
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Get unique models
    models = combined_df["model"].unique().tolist()
    logger.info(f"Models in report: {models}")

    # Calculate statistics per model
    model_stats = {}
    for model in models:
        model_df = combined_df[combined_df["model"] == model]

        total_docs = len(model_df)
        successful = model_df["success"].sum()
        failed = total_docs - successful
        success_rate = (successful / total_docs * 100) if total_docs > 0 else 0.0

        # Token statistics
        successful_df = model_df[model_df["success"]]
        avg_tokens = successful_df["token_count"].mean() if len(successful_df) > 0 else 0.0
        total_tokens = successful_df["token_count"].sum() if len(successful_df) > 0 else 0

        # Error breakdown
        error_df = model_df[~model_df["success"]]
        error_types = {}
        if len(error_df) > 0:
            for error in error_df["error"].dropna():
                error_key = str(error)[:50]  # Truncate long errors
                error_types[error_key] = error_types.get(error_key, 0) + 1

        model_stats[model] = {
            "total_docs": total_docs,
            "successful": int(successful),
            "failed": int(failed),
            "success_rate": success_rate,
            "avg_tokens": avg_tokens,
            "total_tokens": int(total_tokens),
            "error_types": error_types,
        }

    # Prepare data for charts
    model_names = [m.split("/")[-1] for m in models]  # Short names
    success_rates = [model_stats[m]["success_rate"] for m in models]
    avg_tokens_list = [model_stats[m]["avg_tokens"] for m in models]

    # Prepare comparison matrix table
    matrix_rows = []
    for model in models:
        stats = model_stats[model]
        matrix_rows.append(
            {
                "model": model.split("/")[-1],
                "total": stats["total_docs"],
                "successful": stats["successful"],
                "failed": stats["failed"],
                "success_rate": f"{stats['success_rate']:.1f}%",
                "avg_tokens": f"{stats['avg_tokens']:.0f}",
                "total_tokens": f"{stats['total_tokens']:,}",
            }
        )

    # Get sample documents for side-by-side comparison
    sample_docs = combined_df["document_id"].unique()[:5]  # First 5 documents
    comparison_rows = []

    for doc_id in sample_docs:
        doc_results = combined_df[combined_df["document_id"] == doc_id]

        row = {"document_id": str(doc_id).split("/")[-1][:50]}  # Truncate path

        for model in models:
            model_result = doc_results[doc_results["model"] == model]
            if len(model_result) > 0:
                result = model_result.iloc[0]
                if result["success"]:
                    # Truncate text for display
                    text = (
                        result["extracted_text"][:200] + "..."
                        if len(result["extracted_text"]) > 200
                        else result["extracted_text"]
                    )
                    row[model] = text
                else:
                    row[model] = f"❌ Error: {result['error'][:50]}"
            else:
                row[model] = "N/A"

        comparison_rows.append(row)

    doc_models = "".join([f"<th>{m.split('/')[-1]}</th>" for m in models])
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Model Comparison Report</title>
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
                max-width: 1600px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }}
            .header h1 {{
                font-size: 2.8em;
                font-weight: 700;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                margin-bottom: 10px;
            }}
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .section {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }}
            .section-title {{
                font-size: 1.5em;
                font-weight: 600;
                color: #667eea;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
            }}
            .charts {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
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
                height: 350px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }}
            th {{
                background: #667eea;
                color: white;
                padding: 14px;
                text-align: left;
                font-weight: 600;
                position: sticky;
                top: 0;
            }}
            td {{
                padding: 12px 14px;
                border-bottom: 1px solid #eee;
                vertical-align: top;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            .model-name {{
                font-weight: 600;
                color: #667eea;
            }}
            .success {{
                color: #28a745;
                font-weight: 600;
            }}
            .failure {{
                color: #dc3545;
                font-weight: 600;
            }}
            .text-sample {{
                font-family: monospace;
                font-size: 0.85em;
                background: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
                max-height: 100px;
                overflow-y: auto;
                white-space: pre-wrap;
                word-break: break-word;
            }}
            .comparison-table {{
                overflow-x: auto;
            }}
            .summary-badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: 600;
                margin: 2px;
            }}
            .badge-success {{
                background: #d4edda;
                color: #155724;
            }}
            .badge-warning {{
                background: #fff3cd;
                color: #856404;
            }}
            .badge-info {{
                background: #d1ecf1;
                color: #0c5460;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📄 OCR Model Comparison Report</h1>
                <p>Batch Document Processing Analysis</p>
            </div>

            <!-- Comparison Matrix -->
            <div class="section">
                <div class="section-title">Model Comparison Matrix</div>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Total Docs</th>
                            <th>Successful</th>
                            <th>Failed</th>
                            <th>Success Rate</th>
                            <th>Avg Tokens</th>
                            <th>Total Tokens</th>
                        </tr>
                    </thead>
                    <tbody>
                        {
        "".join(
            [
                f'''
                        <tr>
                            <td class="model-name">{row["model"]}</td>
                            <td>{row["total"]}</td>
                            <td class="success">{row["successful"]}</td>
                            <td class="failure">{row["failed"]}</td>
                            <td>{row["success_rate"]}</td>
                            <td>{row["avg_tokens"]}</td>
                            <td>{row["total_tokens"]}</td>
                        </tr>
                        '''
                for row in matrix_rows
            ]
        )
    }
                    </tbody>
                </table>
            </div>

            <!-- Charts -->
            <div class="charts">
                <div class="chart-container">
                    <div class="chart-title">Success Rate Comparison</div>
                    <div class="chart-wrapper">
                        <canvas id="successChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Average Tokens per Model</div>
                    <div class="chart-wrapper">
                        <canvas id="tokensChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Side-by-Side Comparison -->
            <div class="section">
                <div class="section-title">Side-by-Side Text Comparison (Sample Documents)</div>
                <div class="comparison-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Document</th>
                                {doc_models}
                            </tr>
                        </thead>
                        <tbody>
                            {
        "".join(
            [
                f'''
                            <tr>
                                <td class="model-name">{row["document_id"]}</td>
                                {
                    "".join([f'<td><div class="text-sample">{row.get(model, "N/A")}</div></td>' for model in models])
                }
                            </tr>
                            '''
                for row in comparison_rows
            ]
        )
    }
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Error Analysis -->
            <div class="section">
                <div class="section-title">Error Analysis</div>
                {
        "".join(
            [
                f'''
                <div style="margin-bottom: 20px;">
                    <h3 style="color: #667eea; margin-bottom: 10px;">{model.split("/")[-1]}</h3>
                    <p><strong>Total Errors:</strong> {model_stats[model]["failed"]}</p>
                    {
                    '<p><strong>Error Types:</strong></p><ul>'
                    + "".join(
                        [
                            f'<li>{err}: {count} occurrences</li>'
                            for err, count in model_stats[model]["error_types"].items()
                        ]
                    )
                    + '</ul>'
                    if model_stats[model]["error_types"]
                    else '<p>No errors encountered</p>'
                }
                </div>
                '''
                for model in models
            ]
        )
    }
            </div>
        </div>

        <script>
            // Success Rate Comparison Chart
            const successCtx = document.getElementById('successChart').getContext('2d');
            new Chart(successCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(model_names)},
                    datasets: [{{
                        label: 'Success Rate (%)',
                        data: {json.dumps(success_rates)},
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: '#667eea',
                        borderWidth: 2,
                        borderRadius: 6
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Higher is Better'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{
                                callback: function(value) {{
                                    return value + '%';
                                }}
                            }}
                        }}
                    }}
                }}
            }});

            // Average Tokens Chart
            const tokensCtx = document.getElementById('tokensChart').getContext('2d');
            new Chart(tokensCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(model_names)},
                    datasets: [{{
                        label: 'Avg Tokens',
                        data: {json.dumps(avg_tokens_list)},
                        backgroundColor: 'rgba(118, 75, 162, 0.8)',
                        borderColor: '#764ba2',
                        borderWidth: 2,
                        borderRadius: 6
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'More tokens may indicate more detail'
                        }}
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

    logger.info("OCR comparison report generated successfully")


@report_env.task(cache="auto")
async def batch_ocr_comparison_with_report(
    models: list[str] = ["Qwen/Qwen2.5-VL-2B-Instruct", "OpenGVLab/InternVL2_5-2B"],  # noqa
    sample_size: int = 10,
    chunk_size: int = 20,
) -> list[flyte.io.DataFrame]:
    """
    Run multi-model OCR comparison and generate interactive report.

    This is a convenience workflow that:
    1. Runs batch_ocr_comparison from batch_ocr module
    2. Generates comparison report
    3. Returns all results

    Args:
        models: List of model names (as strings for CLI compatibility)
        sample_size: Number of documents
        chunk_size: Chunk size

    Returns:
        List of DataFrames with OCR results per model
    """
    from batch_ocr import OCRModel, batch_ocr_comparison

    # Convert string model names to enum values
    model_enums = []
    for model_str in models:
        # Find matching enum by value
        for model_enum in OCRModel:
            if model_enum.value == model_str:
                model_enums.append(model_enum)
                break

    if not model_enums:
        raise ValueError(f"No valid models found. Provided: {models}")

    logger.info(f"Running comparison with {len(model_enums)} models")

    # Run comparison
    results = await batch_ocr_comparison(
        models=model_enums,
        sample_size=sample_size,
        chunk_size=chunk_size,
    )

    # Generate report
    await generate_ocr_comparison_report(results)

    logger.info("OCR comparison with report completed")
    return results


if __name__ == "__main__":
    from pathlib import Path

    import flyte

    flyte.init_from_config(root_dir=Path(__file__).parent)
    r = flyte.run(batch_ocr_comparison_with_report)
    print(r.url)

    print("""
OCR Comparison Report Generator
================================

This module generates interactive HTML reports comparing multiple OCR models.

USAGE:
------

# Import and use in your workflow:
from batch_ocr_report import generate_ocr_comparison_report

results = await batch_ocr_comparison(...)
await generate_ocr_comparison_report(results)

# Or use the combined workflow:
flyte run batch_ocr_report.py batch_ocr_comparison_with_report \\
    --models='["Qwen/Qwen2.5-VL-2B-Instruct", "OpenGVLab/InternVL2_5-2B"]' \\
    --sample_size=50

REPORT FEATURES:
----------------
✓ Model comparison matrix with statistics
✓ Success rate comparison chart
✓ Token count distribution
✓ Side-by-side text comparisons
✓ Error analysis per model
✓ Interactive Chart.js visualizations
✓ Beautiful gradient UI

The report shows:
- Success rates and failure counts
- Average tokens extracted per model
- Sample document comparisons
- Error breakdowns by type
""")
