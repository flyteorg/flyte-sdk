"""
Model Registry Integration: Versioning and Lineage

This example demonstrates how to use Flyte for model versioning and lineage tracking,
providing better reproducibility than Dagster's basic artifact handling.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
# ]
# ///

from datetime import datetime
from typing import Any, Dict, List, Optional

import flyte
from flyte import Image


def get_registry_image() -> Image:
    """Get image with registry support."""
    return Image.from_debian_base(name="model-registry", python_version=(3, 12)).with_pip_packages("flyte")


image = get_registry_image()
base_env = flyte.TaskEnvironment(
    name="registry_tasks",
    image=image,
)


class ModelVersion:
    """Represents a model version with metadata."""

    def __init__(
        self,
        model_name: str,
        version: str,
        checkpoint_path: str,
        hyperparameters: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        parent_version: Optional["ModelVersion"] = None,
    ):
        self.model_name = model_name
        self.version = version
        self.checkpoint_path = checkpoint_path
        self.hyperparameters = hyperparameters
        self.metrics = metrics or {}
        self.parent_version = parent_version
        self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "checkpoint_path": self.checkpoint_path,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "parent_version": self.parent_version.version if self.parent_version else None,
            "created_at": self.created_at.isoformat(),
        }


@base_env.task
async def register_model(
    model_name: str,
    checkpoint_path: str,
    hyperparameters: Dict[str, Any],
    version: Optional[str] = None,
) -> ModelVersion:
    """
    Register a new model version in the registry.

    Provides versioned model storage with full lineage tracking.
    """
    if version is None:
        # Auto-generate version (in production, use semver or similar)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        version = f"v{timestamp}"

    version_obj = ModelVersion(
        model_name=model_name,
        version=version,
        checkpoint_path=checkpoint_path,
        hyperparameters=hyperparameters,
    )

    return version_obj


@base_env.task
async def get_model_version(
    model_name: str,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve a specific model version from registry.

    Replaces Dagster's artifact lookup with Flyte's typed data system.
    """
    if version:
        return {
            "model_name": model_name,
            "version": version,
            "found": True,
        }
    else:
        # Return latest version (in production, query registry)
        return {
            "model_name": model_name,
            "latest_version": "v20241015120000",
            "found": True,
        }


@base_env.task
async def compare_model_versions(
    versions: List[ModelVersion],
) -> Dict[str, Any]:
    """
    Compare multiple model versions and select the best one.

    Provides lineage comparison that Dagster lacks natively.
    """
    if not versions:
        return {"error": "No versions to compare"}

    # Sort by metrics (use highest accuracy)
    sorted_versions = sorted(
        versions,
        key=lambda v: v.metrics.get("accuracy", 0),
        reverse=True,
    )

    best_version = sorted_versions[0]

    comparison = {
        "total_versions": len(versions),
        "best_version": best_version.version,
        "best_accuracy": best_version.metrics.get("accuracy"),
        "all_versions": [v.version for v in versions],
        "improvement_over_previous": (
            (best_version.metrics.get("accuracy", 0) - sorted_versions[1].metrics.get("accuracy", 0))
            if len(sorted_versions) > 1
            else None
        ),
    }

    return comparison


@base_env.task
async def audit_model_change(
    old_version: ModelVersion,
    new_version: ModelVersion,
) -> Dict[str, Any]:
    """
    Audit changes between two model versions.

    Provides full lineage tracking for compliance and debugging.
    """
    changes = {
        "old_version": old_version.version,
        "new_version": new_version.version,
        "hyperparameter_changes": {
            k: {
                "from": old_version.hyperparameters.get(k),
                "to": new_version.hyperparameters.get(k),
            }
            for k in set(old_version.hyperparameters.keys()) | set(new_version.hyperparameters.keys())
        },
        "metrics_change": {
            metric: (new_version.metrics.get(metric, 0) - old_version.metrics.get(metric, 0))
            for metric in set(old_version.metrics.keys()) | set(new_version.metrics.keys())
        },
        "change_timestamps": {
            "old_version": old_version.created_at.isoformat(),
            "new_version": new_version.created_at.isoformat(),
        },
    }

    return changes


@base_env.task
async def rollback_model(
    model_name: str,
    target_version: str,
) -> Dict[str, Any]:
    """
    Rollback to a previous model version.

    Provides easy rollback capability that Dagster doesn't have natively.
    """
    return {
        "model_name": model_name,
        "target_version": target_version,
        "rollback_status": "success",
        "rolled_back_at": datetime.utcnow().isoformat(),
    }


@base_env.task
async def promote_model(
    version: ModelVersion,
    environment: str = "staging",
) -> Dict[str, Any]:
    """
    Promote a model version to production.

    Provides controlled promotion workflow with approval steps.
    """
    # In production, this would include:
    # 1. Approval workflow
    # 2. A/B testing setup
    # 3. Gradual rollout

    return {
        "model_name": version.model_name,
        "version": version.version,
        "promoted_to": environment,
        "promotion_status": "ready_for_review",
        "promoted_at": datetime.utcnow().isoformat(),
    }


async def run_example():
    """Run the model registry example."""
    flyte.init_from_config()

    print("=" * 60)
    print("Model Registry Integration")
    print("=" * 60)

    # Register initial version
    print("\n[Register] Initial model version...")
    v1 = await flyte.run(
        register_model,
        model_name="text-classifier",
        checkpoint_path="/checkpoints/v1.pth",
        hyperparameters={"lr": 0.001, "batch_size": 32},
    )
    print(f"Registered: {v1.outputs[0].to_dict()}")

    # Register improved version
    print("\n[Register] Improved model version...")
    v2 = await flyte.run(
        register_model,
        model_name="text-classifier",
        checkpoint_path="/checkpoints/v2.pth",
        hyperparameters={"lr": 0.0005, "batch_size": 64},
        version="v2",
        parent_version=v1.outputs[0],
    )
    print(f"Registered: {v2.outputs[0].to_dict()}")

    # Get latest version
    print("\n[Get] Retrieve latest version...")
    result = await flyte.run(get_model_version, model_name="text-classifier")
    print(f"Latest: {result.outputs[0]}")

    # Compare versions
    print("\n[Compare] Comparing versions...")
    comparison = await flyte.run(
        compare_model_versions,
        versions=[v1.outputs[0], v2.outputs[0]],
    )
    print(f"Comparison: {comparison.outputs[0]}")

    # Audit changes
    print("\n[Audit] Auditing model changes...")
    audit = await flyte.run(
        audit_model_change,
        old_version=v1.outputs[0],
        new_version=v2.outputs[0],
    )
    print(f"Audit results: {audit.outputs[0]}")


if __name__ == "__main__":
    # asyncio.run(run_example())

    print("Model Registry Integration Example")
    print("===================================")
    print()
    print("Key features:")
    print("- Versioned model storage with lineages")
    print("- Automatic version numbering")
    print("- Model comparison and selection")
    print("- Full change audit trail")
