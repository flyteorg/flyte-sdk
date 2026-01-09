"""
Configuration for Docker image layer optimization.
"""

HEAVY_DEPENDENCIES = frozenset(
    {
        "tensorflow",
        "torch",
        "torchaudio",
        "torchvision",
        "scikit-learn",
    }
)
