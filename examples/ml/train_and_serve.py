# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "xgboost",
#     "scikit-learn",
#     "pandas",
#     "pyarrow",
#     "joblib",
#     "pydantic",
#     "flyte>=2.0.0b35",
# ]
# ///
"""
Penguin Classification - Train and Serve Example

This script demonstrates a complete ML workflow with two environments:
1. TaskEnvironment: Loads the penguins dataset and trains an XGBoost model
2. FastAPIAppEnvironment: Serves the trained model for inference

The model predicts penguin species based on physical measurements.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import flyte
import flyte.io
from flyte.app import Link, Parameter, RunOutput
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Environment Configuration
# ============================================================================

MODEL_PATH_ENV = "MODEL_PATH"

# Create image from script dependencies
image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_pip_packages("fastapi", "uvicorn", "xgboost", "scikit-learn", "pandas", "pyarrow", "joblib", "pydantic")
)

# Training environment
training_env = flyte.TaskEnvironment(
    name="penguin_training",
    image=image,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    cache="auto",
)


# ============================================================================
# Training Tasks
# ============================================================================


@training_env.task
async def load_penguins_data() -> flyte.io.DataFrame:
    """
    Load the Palmer Penguins dataset.

    Returns:
        DataFrame containing the penguins dataset
    """
    logger.info("Loading penguins dataset...")

    # The penguins dataset (simplified version of Palmer Penguins)
    # Using seaborn's builtin dataset URL
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df = pd.read_csv(url)

    # Drop rows with missing values
    df = df.dropna()

    logger.info(f"Loaded {len(df)} penguin samples with columns: {list(df.columns)}")
    logger.info(f"Species distribution:\n{df['species'].value_counts()}")

    return flyte.io.DataFrame.from_df(df)


@training_env.task
async def train_xgboost_model(
    data: flyte.io.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> flyte.io.File:
    """
    Train an XGBoost classifier on the penguins dataset.

    Args:
        data: DataFrame containing the penguins dataset
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        File containing the trained model saved with joblib
    """
    logger.info("Training XGBoost model...")

    # Load the data
    df = await data.open(pd.DataFrame).all()

    # Feature columns (numeric measurements)
    feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    X = df[feature_cols]
    y = df["species"]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Train XGBoost classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    logger.info(f"Train accuracy: {train_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    # Save model and label encoder together
    model_dir = Path("/tmp/penguin_model")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    model_artifacts = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "classes": label_encoder.classes_.tolist(),
    }
    joblib.dump(model_artifacts, model_path)

    logger.info(f"Model saved to {model_path}")
    return await flyte.io.File.from_local(model_path)


@training_env.task
async def training_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
) -> flyte.io.File:
    """
    Complete training pipeline for the penguin classifier.

    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        File containing the trained model
    """
    logger.info("=== Starting Penguin Classifier Training Pipeline ===")

    # Load data
    data = await load_penguins_data()

    # Train model
    model_file = await train_xgboost_model(
        data=data,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info("=== Training Pipeline Complete ===")
    return model_file


# ============================================================================
# Serving Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    """
    logger.info("Starting up: Loading model...")

    # Get model path from environment or use default
    model_path = Path(os.environ.get(MODEL_PATH_ENV, "/tmp/penguin_model/model.joblib"))

    if model_path.exists():
        artifacts = joblib.load(model_path)
        app.state.model = artifacts["model"]
        app.state.label_encoder = artifacts["label_encoder"]
        app.state.feature_cols = artifacts["feature_cols"]
        app.state.classes = artifacts["classes"]
        logger.info(f"Model loaded successfully. Classes: {app.state.classes}")
    else:
        logger.warning(f"Model not found at {model_path}. App will start but won't be ready.")
        app.state.model = None
        app.state.label_encoder = None
        app.state.feature_cols = None
        app.state.classes = None

    yield

    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Penguin Species Classifier API",
    description="Predict penguin species based on physical measurements using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)

# Create Flyte FastAPI App Environment
serving_env = FastAPIAppEnvironment(
    name="penguin-classifier-api",
    app=app,
    description="Penguin species classification API using XGBoost",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
    parameters=[
        Parameter(
            name="model",
            value=RunOutput(task_name="penguin_training.training_pipeline", type="file"),
            download=True,
            env_var=MODEL_PATH_ENV,
        ),
    ],
    # NOTE: this is a workaround! apps should have this env var auto-injected by the controller
    secrets=[flyte.Secret(key="UNION_API_KEY", as_env_var="_UNION_EAGER_API_KEY")],
    links=[Link(path="/docs", title="Swagger Docs", is_relative=True)],
)


# Request/Response models
class PenguinFeatures(BaseModel):
    """Input features for penguin classification."""

    bill_length_mm: float = Field(..., description="Bill length in millimeters", ge=30, le=60)
    bill_depth_mm: float = Field(..., description="Bill depth in millimeters", ge=13, le=22)
    flipper_length_mm: float = Field(..., description="Flipper length in millimeters", ge=170, le=235)
    body_mass_g: float = Field(..., description="Body mass in grams", ge=2700, le=6500)


class PredictionResponse(BaseModel):
    """Response containing the prediction and probabilities."""

    species: str = Field(..., description="Predicted penguin species")
    confidence: float = Field(..., description="Confidence score for the prediction")
    probabilities: dict[str, float] = Field(..., description="Probability for each species")


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PenguinFeatures):
    """
    Predict penguin species from physical measurements.

    Example input:
    - Adelie: bill_length=39.1, bill_depth=18.7, flipper_length=181, body_mass=3750
    - Gentoo: bill_length=46.1, bill_depth=13.2, flipper_length=211, body_mass=4500
    - Chinstrap: bill_length=46.5, bill_depth=17.9, flipper_length=192, body_mass=3500
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare input
    input_data = pd.DataFrame(
        [[features.bill_length_mm, features.bill_depth_mm, features.flipper_length_mm, features.body_mass_g]],
        columns=app.state.feature_cols,
    )

    # Get prediction and probabilities
    prediction = app.state.model.predict(input_data)[0]
    probabilities = app.state.model.predict_proba(input_data)[0]

    # Decode prediction
    species = app.state.classes[prediction]
    confidence = float(probabilities[prediction])

    # Create probability dict
    prob_dict = {cls: float(prob) for cls, prob in zip(app.state.classes, probabilities)}

    logger.info(f"Prediction: {species} (confidence: {confidence:.4f})")

    return PredictionResponse(
        species=species,
        confidence=confidence,
        probabilities=prob_dict,
    )


@app.get("/classes")
async def get_classes():
    """Get the list of penguin species the model can predict."""
    if app.state.classes is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": app.state.classes}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=Path(__file__).parent,
        log_level=logging.DEBUG,
    )

    # train model
    run = flyte.run(training_pipeline, test_size=0.25, random_state=40)
    run.wait()

    app = flyte.serve(serving_env)
    print(app.url)
