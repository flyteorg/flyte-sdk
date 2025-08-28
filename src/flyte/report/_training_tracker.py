"""
Training Loss Tracker Utility

A reusable utility for visualizing real-time training progress in Flyte tasks.
Provides interactive dashboards with loss curves, metrics tables, and training statistics.
"""

import time
from typing import Any, Dict, List, Optional

import flyte
import flyte.report


class TrainingLossTracker:
    """
    A utility class for tracking and visualizing training progress in real-time.

    Features:
    - Real-time loss curves (training and validation)
    - Metrics table with recent epochs
    - Training statistics (accuracy, learning rate, etc.)
    - Customizable chart colors and styling
    - Support for multiple metrics beyond loss
    """

    def __init__(
        self,
        total_epochs: int,
        title: str = "🚀 Training Progress Dashboard",
        description: str = "Streaming real-time training metrics...",
        metrics_to_track: List[str] = None,
    ):
        self.total_epochs = total_epochs
        self.title = title
        self.description = description
        self.metrics_to_track = metrics_to_track or ["training_loss", "validation_loss", "accuracy", "learning_rate"]

        # Tracking variables
        self.current_epoch = 0
        self.start_time = time.time()
        self.training_losses: List[float] = []
        self.validation_losses: List[float] = []
        self.metrics_history: Dict[str, List[float]] = {metric: [] for metric in self.metrics_to_track}
        self.initialized = False
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    async def initialize(self) -> None:
        """Initialize the training dashboard HTML/CSS/JavaScript components."""
        if self.initialized:
            return

        # Build metrics table headers dynamically
        metric_headers = ""
        for metric in self.metrics_to_track:
            display_name = metric.replace("_", " ").title()
            metric_headers += f"<th>{display_name}</th>"

        await flyte.report.log.aio(
            f"""
        <h1>{self.title}</h1>
        <p>{self.description}</p>
        
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 2;">
                <h3>Training Progress</h3>
                <div id="progress-info">
                    <p><strong>Epoch:</strong> <span id="current-epoch">0</span>/<span id="total-epochs">{self.total_epochs}</span></p>
                    <p><strong>Elapsed Time:</strong> <span id="elapsed-time">0s</span></p>
                    <p><strong>ETA:</strong> <span id="eta">--</span></p>
                </div>
                <div style="background: #e9ecef; height: 20px; border-radius: 10px; position: relative; margin-top: 10px;">
                    <div id="epoch-progress" style="background: linear-gradient(90deg, #007bff, #0056b3); height: 100%; border-radius: 10px; width: 0%; transition: width 0.5s;"></div>
                    <div id="epoch-progress-text" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: bold; color: white; font-size: 12px;">0%</div>
                </div>
            </div>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1;">
                <h3>Best Model</h3>
                <div id="best-model-info">
                    <p><strong>Best Val Loss:</strong> <span id="best-val-loss">--</span></p>
                    <p><strong>Best Epoch:</strong> <span id="best-epoch">--</span></p>
                    <p><strong>Status:</strong> <span id="training-status">Starting...</span></p>
                </div>
            </div>
        </div>
        
        <div style="display: flex; gap: 20px;">
            <div style="flex: 2;">
                <h3>Loss Curves</h3>
                <canvas id="lossChart" width="800" height="400"></canvas>
            </div>
            
            <div style="flex: 1;">
                <h3>Recent Metrics</h3>
                <div style="height: 400px; overflow-y: auto;">
                    <table id="metricsTable" border="1" style="width:100%; border-collapse:collapse; font-size: 12px;">
                        <thead>
                            <tr>
                                <th>Epoch</th>
                                {metric_headers}
                            </tr>
                        </thead>
                        <tbody id="metricsBody">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <script>
            const canvas = document.getElementById('lossChart');
            const ctx = canvas.getContext('2d');
            const trainLosses = [];
            const valLosses = [];
            let bestValLoss = Infinity;
            let bestEpoch = 0;
            
            function drawChart() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw axes
                ctx.strokeStyle = '#666';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(60, 50);
                ctx.lineTo(60, 350);
                ctx.lineTo(750, 350);
                ctx.stroke();
                
                // Draw labels
                ctx.fillStyle = '#666';
                ctx.font = '12px Arial';
                ctx.fillText('Loss', 20, 30);
                ctx.fillText('Epoch', 720, 370);
                
                // Draw grid lines
                ctx.strokeStyle = '#ddd';
                ctx.lineWidth = 1;
                for (let i = 1; i <= 10; i++) {{
                    const y = 50 + (i * 30);
                    ctx.beginPath();
                    ctx.moveTo(60, y);
                    ctx.lineTo(750, y);
                    ctx.stroke();
                }}
                
                // Draw training loss
                if (trainLosses.length > 1) {{
                    ctx.strokeStyle = '#3498db';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    const maxLoss = Math.max(...trainLosses.concat(valLosses));
                    const minLoss = Math.min(...trainLosses.concat(valLosses));
                    const range = maxLoss - minLoss || 1;
                    
                    for (let i = 0; i < trainLosses.length; i++) {{
                        const x = 60 + (i / Math.max(trainLosses.length - 1, 1)) * 690;
                        const y = 350 - ((trainLosses[i] - minLoss) / range) * 300;
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }}
                    ctx.stroke();
                }}
                
                // Draw validation loss
                if (valLosses.length > 1) {{
                    ctx.strokeStyle = '#e74c3c';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    const maxLoss = Math.max(...trainLosses.concat(valLosses));
                    const minLoss = Math.min(...trainLosses.concat(valLosses));
                    const range = maxLoss - minLoss || 1;
                    
                    for (let i = 0; i < valLosses.length; i++) {{
                        const x = 60 + (i / Math.max(valLosses.length - 1, 1)) * 690;
                        const y = 350 - ((valLosses[i] - minLoss) / range) * 300;
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }}
                    ctx.stroke();
                }}
                
                // Draw legend
                ctx.fillStyle = '#3498db';
                ctx.fillRect(600, 70, 15, 10);
                ctx.fillStyle = 'black';
                ctx.font = '11px Arial';
                ctx.fillText('Training Loss', 620, 80);
                
                ctx.fillStyle = '#e74c3c';
                ctx.fillRect(600, 85, 15, 10);
                ctx.fillStyle = 'black';
                ctx.fillText('Validation Loss', 620, 95);
                
                // Draw current values
                if (trainLosses.length > 0) {{
                    ctx.fillStyle = '#333';
                    ctx.font = '10px Arial';
                    ctx.fillText(`Train: ${{trainLosses[trainLosses.length-1].toFixed(4)}}`, 600, 110);
                    if (valLosses.length > 0) {{
                        ctx.fillText(`Val: ${{valLosses[valLosses.length-1].toFixed(4)}}`, 600, 125);
                    }}
                }}
            }}
            
            window.updateTrainingProgress = function(epoch, totalEpochs, trainLoss, valLoss, metrics, elapsedTime) {{
                // Update progress info
                const progress = (epoch / totalEpochs) * 100;
                const eta = elapsedTime > 0 ? Math.ceil((elapsedTime / epoch) * (totalEpochs - epoch)) : 0;
                
                document.getElementById('current-epoch').textContent = epoch;
                document.getElementById('total-epochs').textContent = totalEpochs;
                document.getElementById('elapsed-time').textContent = Math.floor(elapsedTime) + 's';
                document.getElementById('eta').textContent = eta > 0 ? eta + 's' : 'Complete';
                
                document.getElementById('epoch-progress').style.width = progress + '%';
                document.getElementById('epoch-progress-text').textContent = progress.toFixed(1) + '%';
                
                // Update best model info
                if (valLoss < bestValLoss) {{
                    bestValLoss = valLoss;
                    bestEpoch = epoch;
                    document.getElementById('best-val-loss').textContent = valLoss.toFixed(4);
                    document.getElementById('best-epoch').textContent = epoch;
                    document.getElementById('training-status').textContent = 'New best model!';
                    document.getElementById('training-status').style.color = '#28a745';
                }} else {{
                    document.getElementById('training-status').textContent = 'Training...';
                    document.getElementById('training-status').style.color = '#6c757d';
                }}
                
                // Update chart
                trainLosses.push(trainLoss);
                valLosses.push(valLoss);
                drawChart();
            }};
            
            window.addMetricRow = function(epoch, metrics) {{
                const tbody = document.getElementById('metricsBody');
                const row = tbody.insertRow(0);
                
                let rowHtml = `<td>${{epoch}}</td>`;
                for (const [key, value] of Object.entries(metrics)) {{
                    if (typeof value === 'number') {{
                        if (key.includes('loss')) {{
                            rowHtml += `<td>${{value.toFixed(4)}}</td>`;
                        }} else if (key.includes('accuracy')) {{
                            rowHtml += `<td>${{(value * 100).toFixed(2)}}%</td>`;
                        }} else if (key.includes('learning_rate') || key.includes('lr')) {{
                            rowHtml += `<td>${{value.toExponential(2)}}</td>`;
                        }} else {{
                            rowHtml += `<td>${{value.toFixed(4)}}</td>`;
                        }}
                    }} else {{
                        rowHtml += `<td>${{value}}</td>`;
                    }}
                }}
                
                row.innerHTML = rowHtml;
                
                // Keep only last 15 rows
                while (tbody.rows.length > 15) {{
                    tbody.deleteRow(tbody.rows.length - 1);
                }}
            }};
            
            // Initialize
            drawChart();
        </script>
        """,
            do_flush=True,
        )

        self.initialized = True

    async def log_epoch(
        self, epoch: int, training_loss: float, validation_loss: Optional[float] = None, **kwargs
    ) -> None:
        """
        Log metrics for a single epoch.

        Args:
            epoch: Current epoch number
            training_loss: Training loss for this epoch
            validation_loss: Validation loss for this epoch (optional)
            **kwargs: Additional metrics to track (accuracy, learning_rate, etc.)
        """
        if not self.initialized:
            await self.initialize()

        self.current_epoch = epoch
        self.training_losses.append(training_loss)

        if validation_loss is not None:
            self.validation_losses.append(validation_loss)
            # Track best validation loss
            if validation_loss < self.best_val_loss:
                self.best_val_loss = validation_loss
                self.best_epoch = epoch
        else:
            validation_loss = self.validation_losses[-1] if self.validation_losses else 0.0

        # Update metrics history
        metrics = {"training_loss": training_loss, "validation_loss": validation_loss}
        metrics.update(kwargs)

        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(value)

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time

        # Update dashboard
        await flyte.report.log.aio(
            f"""
        <script>
            updateTrainingProgress({epoch}, {self.total_epochs}, {training_loss}, {validation_loss}, {dict(metrics)}, {elapsed_time});
            addMetricRow({epoch}, {dict(metrics)});
        </script>
        """,
            do_flush=True,
        )

    async def log_training_step(self, epoch: int, step: int, loss: float, message: Optional[str] = None) -> None:
        """
        Log a training step (for within-epoch updates).

        Args:
            epoch: Current epoch
            step: Current step within epoch
            loss: Loss at this step
            message: Optional message about this step
        """
        if message:
            await flyte.report.log.aio(
                f"""
            <div style="font-family: monospace; font-size: 11px; color: #6c757d; margin: 2px 0;">
                Epoch {epoch}, Step {step}: {message} (Loss: {loss:.4f})
            </div>
            """,
                do_flush=True,
            )

    async def complete(self, final_message: Optional[str] = None, save_model: bool = True) -> Dict[str, Any]:
        """
        Mark training as complete and show final statistics.

        Args:
            final_message: Optional final completion message
            save_model: Whether to indicate model was saved

        Returns:
            Dictionary with final training statistics
        """
        total_time = time.time() - self.start_time
        final_train_loss = self.training_losses[-1] if self.training_losses else 0.0
        final_val_loss = self.validation_losses[-1] if self.validation_losses else 0.0

        completion_message = final_message or "Training completed successfully!"
        model_status = "Model saved and ready for deployment" if save_model else "Training complete"

        await flyte.report.log.aio(
            f"""
        <script>
            document.getElementById('training-status').textContent = '{model_status}';
            document.getElementById('training-status').style.color = '#28a745';
        </script>
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 20px; border-radius: 8px; margin-top: 20px;">
            <h3>✅ {completion_message}</h3>
            <ul>
                <li><strong>Total Epochs:</strong> {self.current_epoch}/{self.total_epochs}</li>
                <li><strong>Training Time:</strong> {total_time:.1f} seconds</li>
                <li><strong>Final Training Loss:</strong> {final_train_loss:.4f}</li>
                <li><strong>Final Validation Loss:</strong> {final_val_loss:.4f}</li>
                <li><strong>Best Validation Loss:</strong> {self.best_val_loss:.4f} (Epoch {self.best_epoch})</li>
                <li><strong>Avg Time per Epoch:</strong> {total_time / max(self.current_epoch, 1):.2f}s</li>
            </ul>
        </div>
        """,
            do_flush=True,
        )

        return {
            "epochs_completed": self.current_epoch,
            "total_epochs": self.total_epochs,
            "training_time": total_time,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "avg_time_per_epoch": total_time / max(self.current_epoch, 1),
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "metrics_history": self.metrics_history,
        }

    async def early_stopping_check(self, patience: int = 5, min_delta: float = 1e-4) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement

        Returns:
            True if training should stop early
        """
        if len(self.validation_losses) < patience + 1:
            return False

        recent_losses = self.validation_losses[-patience - 1 :]
        best_recent = min(recent_losses[:-1])
        current = recent_losses[-1]

        if current > best_recent - min_delta:
            # No improvement
            await flyte.report.log.aio(
                """
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>⚠️ Early Stopping Triggered:</strong> No improvement in validation loss for the last """
                + str(patience)
                + """ epochs.
            </div>
            """,
                do_flush=True,
            )
            return True

        return False
