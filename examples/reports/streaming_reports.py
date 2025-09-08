import asyncio
import json
import math
import random
import time
from datetime import datetime
from typing import List

import flyte
import flyte.report

env = flyte.TaskEnvironment(name="streaming_reports")

@env.task(report=True)
async def training_loss_visualization(epochs: int = 60) -> str:
    """
    Simulates a training process with streaming loss curve visualization.
    Updates every second for approximately 1 minute.
    """
    await flyte.report.log.aio("""
    <h1>🚀 Training Loss Visualization</h1>
    <p>Streaming real-time training metrics...</p>
    <div id="loss-container">
        <canvas id="lossChart" width="800" height="400"></canvas>
    </div>
    <div id="metrics-table">
        <h3>Training Metrics</h3>
        <table id="metricsTable" border="1" style="width:100%; border-collapse:collapse;">
            <thead>
                <tr>
                    <th>Epoch</th>
                    <th>Training Loss</th>
                    <th>Validation Loss</th>
                    <th>Accuracy</th>
                    <th>Learning Rate</th>
                </tr>
            </thead>
            <tbody id="metricsBody">
            </tbody>
        </table>
    </div>
    <script>
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        const losses = [];
        const valLosses = [];
        
        function drawChart() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw axes
            ctx.beginPath();
            ctx.moveTo(50, 50);
            ctx.lineTo(50, 350);
            ctx.lineTo(750, 350);
            ctx.stroke();
            
            // Draw training loss
            if (losses.length > 1) {
                ctx.strokeStyle = '#3498db';
                ctx.lineWidth = 2;
                ctx.beginPath();
                for (let i = 0; i < losses.length; i++) {
                    const x = 50 + (i / losses.length) * 700;
                    const y = 350 - (losses[i] / 3) * 300;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
            
            // Draw validation loss
            if (valLosses.length > 1) {
                ctx.strokeStyle = '#e74c3c';
                ctx.lineWidth = 2;
                ctx.beginPath();
                for (let i = 0; i < valLosses.length; i++) {
                    const x = 50 + (i / valLosses.length) * 700;
                    const y = 350 - (valLosses[i] / 3) * 300;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
            
            // Draw legend
            ctx.fillStyle = '#3498db';
            ctx.fillRect(600, 70, 20, 10);
            ctx.fillStyle = 'black';
            ctx.fillText('Training Loss', 630, 80);
            
            ctx.fillStyle = '#e74c3c';
            ctx.fillRect(600, 90, 20, 10);
            ctx.fillStyle = 'black';
            ctx.fillText('Validation Loss', 630, 100);
        }
        
        window.updateChart = function(trainLoss, valLoss) {
            losses.push(trainLoss);
            valLosses.push(valLoss);
            drawChart();
        };
        
        window.addMetricRow = function(epoch, trainLoss, valLoss, accuracy, lr) {
            const tbody = document.getElementById('metricsBody');
            const row = tbody.insertRow(0);
            row.innerHTML = `
                <td>${epoch}</td>
                <td>${trainLoss.toFixed(4)}</td>
                <td>${valLoss.toFixed(4)}</td>
                <td>${(accuracy * 100).toFixed(2)}%</td>
                <td>${lr.toExponential(2)}</td>
            `;
            
            // Keep only last 10 rows
            while (tbody.rows.length > 10) {
                tbody.deleteRow(tbody.rows.length - 1);
            }
        };
    </script>
    """, do_flush=True)
    print(f"Training loss visualization started for {epochs} epochs.", flush=True)

    # Simulate training process
    initial_loss = 2.5
    initial_val_loss = 2.7
    learning_rate = 0.001

    for epoch in range(1, epochs + 1):
        # Simulate decreasing loss with some noise
        noise_factor = random.uniform(0.95, 1.05)
        decay_factor = math.exp(-epoch * 0.05)

        train_loss = (initial_loss * decay_factor + 0.1) * noise_factor
        val_loss = (initial_val_loss * decay_factor + 0.15) * noise_factor
        accuracy = min(0.95, 0.3 + (1 - decay_factor) * 0.7)

        # Learning rate decay
        if epoch % 20 == 0:
            learning_rate *= 0.5

        # Update visualization
        await flyte.report.log.aio(f"""
        <script>
            updateChart({train_loss}, {val_loss});
            addMetricRow({epoch}, {train_loss}, {val_loss}, {accuracy}, {learning_rate});
        </script>
        <p><strong>Epoch {epoch}/{epochs}</strong> - Training Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy*100:.2f}%</p>
        """, do_flush=True)

        print(f"Training loss visualization started for {epochs} epochs.", flush=True)
        await asyncio.sleep(1)  # Update every second

    await flyte.report.log.aio("""
    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <h3>✅ Training Completed Successfully!</h3>
        <p>Final model achieved excellent convergence with validation accuracy above 90%.</p>
    </div>
    """, do_flush=True)

    print("Training visualization completed successfully.", flush=True)
    return "Training visualization completed"


@env.task(report=True)
async def data_processing_dashboard(total_records: int = 50000) -> str:
    """
    Simulates a data processing pipeline with real-time progress visualization.
    Updates every second for approximately 1 minute.
    """
    await flyte.report.log.aio("""
    <h1>📊 Data Processing Dashboard</h1>
    <p>Processing records in real-time...</p>
    
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1;">
            <h3>Progress Overview</h3>
            <div id="progress-container">
                <div style="background: #e9ecef; height: 30px; border-radius: 15px; position: relative;">
                    <div id="progress-bar" style="background: linear-gradient(90deg, #28a745, #20c997); height: 100%; border-radius: 15px; width: 0%; transition: width 0.5s;"></div>
                    <div id="progress-text" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: bold; color: white;">0%</div>
                </div>
            </div>
        </div>
        
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1;">
            <h3>Processing Stats</h3>
            <div id="stats">
                <p>Records Processed: <span id="processed">0</span></p>
                <p>Success Rate: <span id="success-rate">0%</span></p>
                <p>Processing Speed: <span id="speed">0</span> records/sec</p>
                <p>Estimated Time Remaining: <span id="eta">--</span></p>
            </div>
        </div>
    </div>
    
    <div style="display: flex; gap: 20px;">
        <div style="flex: 2;">
            <h3>Processing Rate (Records/Second)</h3>
            <canvas id="rateChart" width="600" height="300"></canvas>
        </div>
        
        <div style="flex: 1;">
            <h3>Recent Activity</h3>
            <div id="activity-log" style="height: 300px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px;">
            </div>
        </div>
    </div>
    
    <script>
        const rateCanvas = document.getElementById('rateChart');
        const rateCtx = rateCanvas.getContext('2d');
        const rates = [];
        let startTime = Date.now();
        
        function drawRateChart() {
            rateCtx.clearRect(0, 0, rateCanvas.width, rateCanvas.height);
            
            // Draw axes
            rateCtx.strokeStyle = '#666';
            rateCtx.beginPath();
            rateCtx.moveTo(50, 50);
            rateCtx.lineTo(50, 250);
            rateCtx.lineTo(550, 250);
            rateCtx.stroke();
            
            // Draw rate line
            if (rates.length > 1) {
                rateCtx.strokeStyle = '#17a2b8';
                rateCtx.lineWidth = 2;
                rateCtx.beginPath();
                
                const maxRate = Math.max(...rates);
                const minRate = Math.min(...rates);
                const range = maxRate - minRate || 1;
                
                for (let i = 0; i < rates.length; i++) {
                    const x = 50 + (i / Math.max(rates.length - 1, 1)) * 500;
                    const y = 250 - ((rates[i] - minRate) / range) * 200;
                    if (i === 0) rateCtx.moveTo(x, y);
                    else rateCtx.lineTo(x, y);
                }
                rateCtx.stroke();
            }
        }
        
        window.updateDashboard = function(processed, total, rate, successRate) {
            const percentage = (processed / total) * 100;
            const eta = rate > 0 ? Math.ceil((total - processed) / rate) : 0;
            
            // Update progress bar
            document.getElementById('progress-bar').style.width = percentage + '%';
            document.getElementById('progress-text').textContent = percentage.toFixed(1) + '%';
            
            // Update stats
            document.getElementById('processed').textContent = processed.toLocaleString();
            document.getElementById('success-rate').textContent = successRate.toFixed(1) + '%';
            document.getElementById('speed').textContent = rate;
            document.getElementById('eta').textContent = eta > 0 ? eta + 's' : 'Complete';
            
            // Update rate chart
            rates.push(rate);
            if (rates.length > 30) rates.shift(); // Keep last 30 points
            drawRateChart();
        };
        
        window.addActivity = function(message) {
            const log = document.getElementById('activity-log');
            const timestamp = new Date().toLocaleTimeString();
            log.innerHTML = `<div>[${timestamp}] ${message}</div>` + log.innerHTML;
            
            // Keep only last 20 entries
            const entries = log.children;
            while (entries.length > 20) {
                log.removeChild(entries[entries.length - 1]);
            }
        };
    </script>
    """, do_flush=True)

    # Simulate data processing
    processed = 0
    errors = 0
    batch_sizes = [800, 850, 900, 950, 1000, 1050, 1100]  # Variable processing rates

    start_time = time.time()

    while processed < total_records:
        # Simulate variable processing speed
        batch_size = random.choice(batch_sizes)

        # Add some processing delays occasionally
        if random.random() < 0.1:  # 10% chance of slower batch
            batch_size = int(batch_size * 0.6)
            await flyte.report.log.aio("""
            <script>addActivity("⚠️ Detected slow processing batch, optimizing...");</script>
            """, do_flush=True)
        elif random.random() < 0.05:  # 5% chance of error
            errors += random.randint(1, 5)
            await flyte.report.log.aio("""
            <script>addActivity("❌ Processing errors detected, retrying failed records...");</script>
            """, do_flush=True)
        else:
            await flyte.report.log.aio(f"""
            <script>addActivity("✅ Successfully processed batch of {batch_size} records");</script>
            """, do_flush=True)

        processed = min(processed + batch_size, total_records)
        current_time = time.time()
        elapsed = current_time - start_time
        rate = int(batch_size) if elapsed < 1 else int(processed / elapsed)
        success_rate = ((processed - errors) / processed) * 100 if processed > 0 else 100

        # Update dashboard
        await flyte.report.log.aio(f"""
        <script>
            updateDashboard({processed}, {total_records}, {rate}, {success_rate});
        </script>
        """, do_flush=True)

        print(f"Processed {processed:,} records, Errors: {errors}, Rate: {rate:,}"
              f" records/sec, Success Rate: {success_rate:.2f}%", flush=True)
        await asyncio.sleep(1)  # Update every second

        if processed >= total_records:
            break

    # Final completion message
    total_time = time.time() - start_time
    avg_rate = int(total_records / total_time)

    await flyte.report.log.aio(f"""
    <script>addActivity("🎉 Processing completed successfully!");</script>
    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 20px; border-radius: 8px; margin-top: 20px;">
        <h3>🎉 Processing Complete!</h3>
        <ul>
            <li><strong>Total Records:</strong> {total_records:,}</li>
            <li><strong>Processing Time:</strong> {total_time:.1f} seconds</li>
            <li><strong>Average Rate:</strong> {avg_rate:,} records/second</li>
            <li><strong>Success Rate:</strong> {success_rate:.2f}%</li>
            <li><strong>Errors Handled:</strong> {errors}</li>
        </ul>
    </div>
    """, do_flush=True)
    print(f"Data processing completed: {processed:,} records processed with {errors} errors.", flush=True)

    return f"Processed {total_records:,} records successfully"


@env.task
async def main():
    """
    Main task to run both reports.
    """
    await asyncio.gather(*[training_loss_visualization(epochs=60), data_processing_dashboard(total_records=50000)])


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(main)
    print(f"Run Name: {run.name}", flush=True)
    print(f"Run URL: {run.url}", flush=True)
