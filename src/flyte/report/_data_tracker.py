"""
Data Processing Tracker Utility

A reusable utility for visualizing real-time data processing progress in Flyte tasks.
Provides interactive dashboards with progress bars, rate charts, and activity logs.
"""

import time
from typing import Any, Dict, List, Optional

import flyte
import flyte.report


class DataProcessingTracker:
    """
    A utility class for tracking and visualizing data processing progress in real-time.

    Features:
    - Progress bar with percentage completion
    - Processing rate chart with historical data
    - Real-time statistics (speed, ETA, success rate)
    - Activity log with timestamped entries
    - Error tracking and reporting
    """

    def __init__(
        self,
        total_records: int,
        title: str = "📊 Data Processing Dashboard",
        description: str = "Processing records in real-time...",
    ):
        self.total_records = total_records
        self.title = title
        self.description = description

        # Tracking variables
        self.processed = 0
        self.errors = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.rates: List[int] = []
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize the dashboard HTML/CSS/JavaScript components."""
        if self.initialized:
            return

        await flyte.report.log.aio(
            f"""
        <h1>{self.title}</h1>
        <p>{self.description}</p>
        
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
                    <p>Success Rate: <span id="success-rate">100%</span></p>
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
            
            function drawRateChart() {{
                rateCtx.clearRect(0, 0, rateCanvas.width, rateCanvas.height);
                
                // Draw axes
                rateCtx.strokeStyle = '#666';
                rateCtx.beginPath();
                rateCtx.moveTo(50, 50);
                rateCtx.lineTo(50, 250);
                rateCtx.lineTo(550, 250);
                rateCtx.stroke();
                
                // Draw labels
                rateCtx.fillStyle = '#666';
                rateCtx.font = '12px Arial';
                rateCtx.fillText('Records/sec', 10, 30);
                rateCtx.fillText('Time', 520, 270);
                
                // Draw rate line
                if (rates.length > 1) {{
                    rateCtx.strokeStyle = '#17a2b8';
                    rateCtx.lineWidth = 2;
                    rateCtx.beginPath();
                    
                    const maxRate = Math.max(...rates);
                    const minRate = Math.min(...rates);
                    const range = maxRate - minRate || 1;
                    
                    for (let i = 0; i < rates.length; i++) {{
                        const x = 50 + (i / Math.max(rates.length - 1, 1)) * 500;
                        const y = 250 - ((rates[i] - minRate) / range) * 200;
                        if (i === 0) rateCtx.moveTo(x, y);
                        else rateCtx.lineTo(x, y);
                    }}
                    rateCtx.stroke();
                    
                    // Draw current rate
                    rateCtx.fillStyle = '#17a2b8';
                    rateCtx.fillText(`Current: ${{rates[rates.length-1]}}`, 460, 30);
                    rateCtx.fillText(`Max: ${{maxRate}}`, 460, 45);
                }}
            }}
            
            window.updateDashboard = function(processed, total, rate, successRate) {{
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
                if (rates.length > 50) rates.shift(); // Keep last 50 points
                drawRateChart();
            }};
            
            window.addActivity = function(message, type = 'info') {{
                const log = document.getElementById('activity-log');
                const timestamp = new Date().toLocaleTimeString();
                const icon = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : type === 'success' ? '✅' : 'ℹ️';
                log.innerHTML = `<div style="margin-bottom: 5px;">[${{timestamp}}] ${{icon}} ${{message}}</div>` + log.innerHTML;
                
                // Keep only last 30 entries
                const entries = log.children;
                while (entries.length > 30) {{
                    log.removeChild(entries[entries.length - 1]);
                }}
            }};
            
            // Initialize
            drawRateChart();
        </script>
        """,
            do_flush=True,
        )

        self.initialized = True
        await self.log_activity("Dashboard initialized", "success")

    async def update_progress(
        self, records_processed: int, errors: int = 0, custom_message: Optional[str] = None
    ) -> None:
        """
        Update the progress dashboard with new data.

        Args:
            records_processed: Number of records processed so far
            errors: Number of errors encountered
            custom_message: Optional custom message to log
        """
        if not self.initialized:
            await self.initialize()

        self.processed = min(records_processed, self.total_records)
        self.errors = errors

        # Calculate rates
        current_time = time.time()
        elapsed = current_time - self.start_time
        rate = int(self.processed / elapsed) if elapsed > 0 else 0

        success_rate = ((self.processed - self.errors) / max(self.processed, 1)) * 100

        # Update dashboard
        await flyte.report.log.aio(
            f"""
        <script>
            updateDashboard({self.processed}, {self.total_records}, {rate}, {success_rate});
        </script>
        """,
            do_flush=True,
        )

        # Log custom message if provided
        if custom_message:
            await self.log_activity(custom_message)

        self.last_update = current_time

    async def log_activity(self, message: str, activity_type: str = "info") -> None:
        """
        Log an activity message to the dashboard.

        Args:
            message: The message to log
            activity_type: Type of message ('info', 'success', 'warning', 'error')
        """
        await flyte.report.log.aio(
            f"""
        <script>addActivity("{message}", "{activity_type}");</script>
        """,
            do_flush=True,
        )

    async def batch_update(self, batch_size: int, batch_errors: int = 0, batch_message: Optional[str] = None) -> None:
        """
        Update progress by a batch amount.

        Args:
            batch_size: Number of records in this batch
            batch_errors: Number of errors in this batch
            batch_message: Optional message for this batch
        """
        new_processed = self.processed + batch_size
        new_errors = self.errors + batch_errors

        if batch_message is None:
            if batch_errors > 0:
                batch_message = f"Processed batch of {batch_size} records ({batch_errors} errors)"
                activity_type = "warning"
            else:
                batch_message = f"Successfully processed batch of {batch_size} records"
                activity_type = "success"
        else:
            activity_type = "error" if batch_errors > 0 else "info"

        await self.update_progress(new_processed, new_errors)
        await self.log_activity(batch_message, activity_type)

    async def complete(self, final_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Mark processing as complete and show final statistics.

        Args:
            final_message: Optional final completion message

        Returns:
            Dictionary with final processing statistics
        """
        total_time = time.time() - self.start_time
        avg_rate = int(self.total_records / total_time) if total_time > 0 else 0
        success_rate = ((self.processed - self.errors) / max(self.processed, 1)) * 100

        completion_message = final_message or "🎉 Processing completed successfully!"

        await flyte.report.log.aio(
            f"""
        <script>addActivity("{completion_message}", "success");</script>
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 20px; border-radius: 8px; margin-top: 20px;">
            <h3>🎉 Processing Complete!</h3>
            <ul>
                <li><strong>Total Records:</strong> {self.total_records:,}</li>
                <li><strong>Records Processed:</strong> {self.processed:,}</li>
                <li><strong>Processing Time:</strong> {total_time:.1f} seconds</li>
                <li><strong>Average Rate:</strong> {avg_rate:,} records/second</li>
                <li><strong>Success Rate:</strong> {success_rate:.2f}%</li>
                <li><strong>Errors Handled:</strong> {self.errors}</li>
            </ul>
        </div>
        """,
            do_flush=True,
        )

        return {
            "total_records": self.total_records,
            "processed": self.processed,
            "processing_time": total_time,
            "average_rate": avg_rate,
            "success_rate": success_rate,
            "errors": self.errors,
        }
