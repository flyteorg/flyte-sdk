"""
Demonstration of the reusable utility modules for tracking data processing and training progress.

This example shows how to use both the DataProcessingTracker and TrainingLossTracker
utilities in your own Flyte tasks.
"""

import asyncio
import math
import random
from typing import Dict, Tuple

import flyte
from flyte.report import DataProcessingTracker, TrainingLossTracker

env = flyte.TaskEnvironment(name="utility_demo")


@env.task(report=True)
async def demo_data_processing(total_records: int = 10000) -> Dict[str, str]:
    """
    Demonstrates the DataProcessingTracker utility with simulated data processing.
    """
    # Initialize the tracker
    tracker = DataProcessingTracker(
        total_records=total_records,
        title="🗂️ Data Processing Demo",
        description="Demonstrating real-time data processing visualization..."
    )
    
    # Initialize the dashboard
    await tracker.initialize()
    
    # Simulate processing with variable batch sizes and occasional errors
    processed = 0
    errors = 0
    batch_sizes = [50, 75, 100, 125, 150, 200]
    
    while processed < total_records:
        # Variable processing speed
        batch_size = random.choice(batch_sizes)
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Simulate occasional errors (5% chance)
        batch_errors = 0
        if random.random() < 0.05:
            batch_errors = random.randint(1, 3)
        
        # Simulate slow batches (10% chance)
        if random.random() < 0.1:
            batch_size = int(batch_size * 0.3)
            await tracker.log_activity("⚠️ Processing bottleneck detected, reducing batch size", "warning")
            await asyncio.sleep(1)  # Extra delay for slow batch
        
        # Update progress
        await tracker.batch_update(
            batch_size=batch_size,
            batch_errors=batch_errors
        )
        
        processed = min(processed + batch_size, total_records)
        errors += batch_errors
    
    # Complete processing
    stats = await tracker.complete("Data processing pipeline completed successfully!")
    
    return {
        "status": "completed",
        "records_processed": str(stats["processed"]),
        "processing_time": f"{stats['processing_time']:.2f}s",
        "average_rate": f"{stats['average_rate']} records/sec",
        "success_rate": f"{stats['success_rate']:.2f}%"
    }


@env.task(report=True)
async def demo_training_progress(epochs: int = 30) -> Dict[str, str]:
    """
    Demonstrates the TrainingLossTracker utility with simulated model training.
    """
    # Initialize the tracker
    tracker = TrainingLossTracker(
        total_epochs=epochs,
        title="🤖 Model Training Demo",
        description="Demonstrating real-time training progress visualization...",
        metrics_to_track=["training_loss", "validation_loss", "accuracy", "learning_rate", "f1_score"]
    )
    
    # Initialize the dashboard
    await tracker.initialize()
    
    # Simulate training with realistic loss curves
    initial_train_loss = 2.5
    initial_val_loss = 2.8
    learning_rate = 0.001
    
    for epoch in range(1, epochs + 1):
        # Simulate training step delays
        await asyncio.sleep(0.8)
        
        # Simulate decreasing loss with noise
        noise_factor = random.uniform(0.95, 1.08)
        decay_factor = math.exp(-epoch * 0.08)
        
        # Training loss (generally decreasing)
        train_loss = (initial_train_loss * decay_factor + 0.05) * noise_factor
        
        # Validation loss (more volatile, sometimes increases)
        val_noise = random.uniform(0.92, 1.12)
        val_loss = (initial_val_loss * decay_factor + 0.08) * val_noise
        
        # Accuracy (improving over time)
        accuracy = min(0.95, 0.2 + (1 - decay_factor) * 0.8 + random.uniform(-0.02, 0.02))
        
        # F1 Score (similar to accuracy but slightly lower)
        f1_score = accuracy * random.uniform(0.95, 1.02)
        
        # Learning rate decay every 10 epochs
        if epoch % 10 == 0 and epoch > 10:
            learning_rate *= 0.5
            await tracker.log_training_step(
                epoch, 0, train_loss, 
                f"Learning rate reduced to {learning_rate:.6f}"
            )
        
        # Log the epoch
        await tracker.log_epoch(
            epoch=epoch,
            training_loss=train_loss,
            validation_loss=val_loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            f1_score=f1_score
        )
        
        # Simulate within-epoch logging for some epochs
        if epoch % 5 == 0:
            await tracker.log_training_step(
                epoch, 100, train_loss, 
                "Completed batch processing"
            )
        
        # Check for early stopping (optional)
        if epoch > 10 and await tracker.early_stopping_check(patience=5):
            break
    
    # Complete training
    stats = await tracker.complete("Model training completed with excellent convergence!")
    
    return {
        "status": "completed",
        "epochs_completed": str(stats["epochs_completed"]),
        "training_time": f"{stats['training_time']:.2f}s",
        "final_train_loss": f"{stats['final_train_loss']:.4f}",
        "final_val_loss": f"{stats['final_val_loss']:.4f}",
        "best_val_loss": f"{stats['best_val_loss']:.4f}",
        "best_epoch": str(stats["best_epoch"])
    }


@env.task(report=True)
async def combined_ml_pipeline(
    dataset_size: int = 5000,
    training_epochs: int = 20
) -> Dict[str, str]:
    """
    Demonstrates using both utilities in a complete ML pipeline.
    """
    # Phase 1: Data Processing
    print("Starting data preprocessing phase...")
    
    data_tracker = DataProcessingTracker(
        total_records=dataset_size,
        title="📊 ML Pipeline: Data Preprocessing",
        description="Loading and preprocessing training data..."
    )
    
    await data_tracker.initialize()
    
    # Simulate data loading and preprocessing
    for batch in range(0, dataset_size, 200):
        batch_size = min(200, dataset_size - batch)
        
        # Simulate preprocessing steps
        await asyncio.sleep(0.3)
        
        # Simulate occasional data quality issues
        batch_errors = random.randint(0, 2) if random.random() < 0.1 else 0
        
        if batch_errors > 0:
            await data_tracker.log_activity(f"Data quality issues found, cleaning {batch_errors} records", "warning")
        
        await data_tracker.batch_update(batch_size, batch_errors)
    
    data_stats = await data_tracker.complete("Data preprocessing completed! Starting model training...")
    
    # Small delay between phases
    await asyncio.sleep(2)
    
    # Phase 2: Model Training
    print("Starting model training phase...")
    
    training_tracker = TrainingLossTracker(
        total_epochs=training_epochs,
        title="🚀 ML Pipeline: Model Training",
        description="Training neural network on preprocessed data...",
        metrics_to_track=["training_loss", "validation_loss", "accuracy", "precision", "recall"]
    )
    
    await training_tracker.initialize()
    
    # Simulate realistic training
    base_train_loss = 1.8
    base_val_loss = 2.0
    lr = 0.01
    
    for epoch in range(1, training_epochs + 1):
        await asyncio.sleep(0.6)
        
        # Realistic loss curves
        progress = epoch / training_epochs
        decay = math.exp(-progress * 3)
        
        train_loss = (base_train_loss * decay + 0.1) * random.uniform(0.98, 1.02)
        val_loss = (base_val_loss * decay + 0.15) * random.uniform(0.95, 1.05)
        
        accuracy = min(0.92, 0.6 + progress * 0.35 + random.uniform(-0.01, 0.01))
        precision = accuracy * random.uniform(0.98, 1.02)
        recall = accuracy * random.uniform(0.96, 1.04)
        
        # Learning rate schedule
        if epoch in [8, 15]:
            lr *= 0.3
            await training_tracker.log_training_step(epoch, 0, train_loss, f"Learning rate adjusted to {lr:.6f}")
        
        await training_tracker.log_epoch(
            epoch=epoch,
            training_loss=train_loss,
            validation_loss=val_loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            learning_rate=lr
        )
    
    training_stats = await training_tracker.complete("ML pipeline completed successfully!")
    
    return {
        "pipeline_status": "completed",
        "data_processing_time": f"{data_stats['processing_time']:.2f}s",
        "data_success_rate": f"{data_stats['success_rate']:.2f}%",
        "training_time": f"{training_stats['training_time']:.2f}s",
        "final_accuracy": f"{training_stats['metrics_history']['accuracy'][-1]*100:.2f}%",
        "best_val_loss": f"{training_stats['best_val_loss']:.4f}"
    }


@env.task
async def main() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Main task to run all demonstrations.
    """
    results = await asyncio.gather(
        demo_data_processing(total_records=8000),
        demo_training_progress(epochs=25),
        combined_ml_pipeline(dataset_size=6000, training_epochs=15)
    )
    
    print("All utility demonstrations completed successfully!")
    return results


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    
    print("🚀 Starting utility module demonstrations...")
    print("This will showcase both DataProcessingTracker and TrainingLossTracker utilities")
    
    run = flyte.run(main)
    print(f"Demo Run URL: {run.url}")
    
    # Wait for completion
    result = run.wait()
    if result:
        print("✅ All demonstrations completed successfully!")
    else:
        print("❌ Some demonstrations failed")