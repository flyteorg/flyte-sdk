from lib_transforms.ops import normalize


def train_mean_predictor(features: list[float], labels: list[float]) -> dict:
    """Train a trivial mean predictor. Returns model params."""
    normalized = normalize(features)
    mean_label = sum(labels) / len(labels) if labels else 0.0
    return {"type": "mean_predictor", "mean": mean_label, "n_features": len(normalized)}


def predict(model: dict, features: list[float]) -> float:
    """Predict using the mean predictor."""
    return model["mean"]
