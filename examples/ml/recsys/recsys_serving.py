# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "sentence-transformers",
#     "numpy",
#     "pydantic",
#     "pandas",
#     "pyarrow",
#     "flyte>=2.0.0b34",
# ]
# ///
"""
Recommendation System - FastAPI Serving Application

This script demonstrates:
1. Loading trained embedding models
2. Serving recommendations via FastAPI endpoints
3. Async request handling for scalability
4. Similarity-based ranking

API Endpoints:
- GET /health - Health check
- POST /recommend - Get item recommendations for a user
- POST /similar-items - Find similar items to a given item
- POST /embed-text - Get embedding for arbitrary text
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

import flyte
import flyte.io
from flyte.app import Input
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    """
    # Startup: Load model and data
    logger.info("Starting up: Loading artifacts...")

    # Get artifacts from environment inputs
    # In production, this would come from the deployed app's inputs
    # For local testing, you can set a path here
    artifacts_path = Path("/tmp/recsys_artifacts")  # Default for local testing

    if artifacts_path.exists():
        await load_model_and_data(artifacts_path)
        logger.info("Startup complete: Model and data loaded")
    else:
        logger.warning(f"Artifacts not found at {artifacts_path}. App will start but won't be ready.")

    yield

    # Shutdown: Clean up resources if needed
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Recommendation System API",
    description="Embedding-based recommendation service",
    version="1.0.0",
    lifespan=lifespan,
)

# Create Flyte FastAPI App Environment
image = flyte.Image.from_uv_script(__file__, name="recsys-serving")

env = FastAPIAppEnvironment(
    name="recsys-fastapi-app",
    app=app,
    description="Recommendation system serving API with embedding-based recommendations",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    requires_auth=False,
    inputs=[
        Input(
            name="artifacts",
            value=flyte.io.Dir.from_existing_remote(os.environ.get("ARTIFACTS_DIR", "/tmp/recsys_artifacts")),
            mount="/tmp/recsys_artifacts",
        )
    ],
)

# Application state stored in app.state instead of globals
# Initialized here to provide type hints and defaults
app.state.model: Optional[SentenceTransformer] = None
app.state.user_embeddings: dict = {}
app.state.item_embeddings: dict = {}
app.state.users_metadata: dict = {}
app.state.items_metadata: dict = {}


# Request/Response models
class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="User ID to get recommendations for")
    top_k: int = Field(10, description="Number of recommendations to return", ge=1, le=100)
    exclude_items: List[str] = Field(default_factory=list, description="Item IDs to exclude from recommendations")


class ItemScore(BaseModel):
    item_id: str
    score: float
    title: str
    category: str
    description: str


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[ItemScore]


class SimilarItemsRequest(BaseModel):
    item_id: str = Field(..., description="Item ID to find similar items for")
    top_k: int = Field(10, description="Number of similar items to return", ge=1, le=100)


class SimilarItemsResponse(BaseModel):
    item_id: str
    similar_items: List[ItemScore]


class EmbedTextRequest(BaseModel):
    text: str = Field(..., description="Text to embed")


class EmbedTextResponse(BaseModel):
    embedding: List[float]
    dimension: int


async def load_model_and_data(artifacts_dir: Path):
    """
    Load the trained model and pre-computed embeddings from parquet files.

    Args:
        artifacts_dir: Path to directory containing all artifacts
    """
    logger.info("Loading model...")
    model_path = artifacts_dir / "model"
    app.state.model = await asyncio.to_thread(SentenceTransformer, str(model_path))

    logger.info("Loading user embeddings from parquet...")
    user_df = pd.read_parquet(artifacts_dir / "user_embeddings.parquet")
    # Convert DataFrame to dictionaries for fast lookup
    app.state.user_embeddings = {row["user_id"]: row["embedding"] for _, row in user_df.iterrows()}
    app.state.users_metadata = {
        row["user_id"]: {
            "user_id": row["user_id"],
            "age_group": row["age_group"],
            "interests": row["interests"],
            "profile": row["profile"],
        }
        for _, row in user_df.iterrows()
    }

    logger.info("Loading item embeddings from parquet...")
    item_df = pd.read_parquet(artifacts_dir / "item_embeddings.parquet")
    # Convert DataFrame to dictionaries for fast lookup
    app.state.item_embeddings = {row["item_id"]: row["embedding"] for _, row in item_df.iterrows()}
    app.state.items_metadata = {
        row["item_id"]: {
            "item_id": row["item_id"],
            "category": row["category"],
            "title": row["title"],
            "description": row["description"],
        }
        for _, row in item_df.iterrows()
    }

    logger.info(
        f"Model and data loaded successfully: "
        f"{len(app.state.user_embeddings)} users, {len(app.state.item_embeddings)} items"
    )


async def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(similarity)


async def rank_items_for_user(
    user_embedding: List[float],
    exclude_items: List[str] | None = None,
    top_k: int = 10,
) -> List[ItemScore]:
    """
    Rank all items for a user based on embedding similarity.

    Args:
        user_embedding: User's embedding vector
        exclude_items: Item IDs to exclude from ranking
        top_k: Number of top items to return

    Returns:
        List of top-k ranked items with scores
    """
    exclude_set = set(exclude_items or [])

    # Compute similarities in parallel
    similarities = []
    for item_id, item_embedding in app.state.item_embeddings.items():
        if item_id in exclude_set:
            continue

        similarity = await compute_similarity(user_embedding, item_embedding)
        similarities.append((item_id, similarity))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Take top-k and create response
    results = []
    for item_id, score in similarities[:top_k]:
        item = app.state.items_metadata[item_id]
        results.append(
            ItemScore(
                item_id=item_id,
                score=score,
                title=item["title"],
                category=item["category"],
                description=item["description"],
            )
        )

    return results


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy" if app.state.model is not None else "not_ready",
        "model_loaded": app.state.model is not None,
        "num_users": len(app.state.user_embeddings),
        "num_items": len(app.state.item_embeddings),
    }


@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    Get personalized item recommendations for a user.

    Returns items ranked by similarity to the user's embedding.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.user_id not in app.state.user_embeddings:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")

    logger.info(f"Getting recommendations for user {request.user_id}")

    # Get user embedding
    user_embedding = app.state.user_embeddings[request.user_id]

    # Rank items
    recommendations = await rank_items_for_user(
        user_embedding=user_embedding,
        exclude_items=request.exclude_items,
        top_k=request.top_k,
    )

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
    )


@app.post("/similar-items", response_model=SimilarItemsResponse)
async def find_similar_items(request: SimilarItemsRequest):
    """
    Find items similar to a given item.

    Useful for "customers who viewed this also viewed" features.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.item_id not in app.state.item_embeddings:
        raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")

    logger.info(f"Finding similar items for {request.item_id}")

    # Get item embedding
    item_embedding = app.state.item_embeddings[request.item_id]

    # Compute similarities to all other items
    similarities = []
    for other_item_id, other_embedding in app.state.item_embeddings.items():
        if other_item_id == request.item_id:
            continue

        similarity = await compute_similarity(item_embedding, other_embedding)
        similarities.append((other_item_id, similarity))

    # Sort and take top-k
    similarities.sort(key=lambda x: x[1], reverse=True)

    similar_items = []
    for item_id, score in similarities[: request.top_k]:
        item = app.state.items_metadata[item_id]
        similar_items.append(
            ItemScore(
                item_id=item_id,
                score=score,
                title=item["title"],
                category=item["category"],
                description=item["description"],
            )
        )

    return SimilarItemsResponse(
        item_id=request.item_id,
        similar_items=similar_items,
    )


@app.post("/embed-text", response_model=EmbedTextResponse)
async def embed_text(request: EmbedTextRequest):
    """
    Generate embedding for arbitrary text.

    Useful for creating embeddings for new users/items on the fly.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Embedding text: {request.text[:50]}...")

    # Generate embedding
    embedding = await asyncio.to_thread(app.state.model.encode, request.text)

    return EmbedTextResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding),
    )


@app.get("/users/{user_id}")
async def get_user_info(user_id: str):
    """
    Get user metadata.
    """
    if user_id not in app.state.users_metadata:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    return app.state.users_metadata[user_id]


@app.get("/items/{item_id}")
async def get_item_info(item_id: str):
    """
    Get item metadata.
    """
    if item_id not in app.state.items_metadata:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    return app.state.items_metadata[item_id]


if __name__ == "__main__":
    import argparse
    import flyte.remote

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", type=str)
    args = parser.parse_args()

    flyte.init_from_config(
        root_dir=Path(__file__).parent,
        log_level=logging.DEBUG,
    )

    run = flyte.remote.Run.get(args.training_run)
    artifacts_dir, *_ = run.outputs()

    # Deploy the FastAPI app to Flyte
    # Note: You need to provide the artifacts directory from the training pipeline
    # You can get this by running the training pipeline first:
    #
    # `uv run recsys_training.py`
    #
    # Then supplying the run name to the serving script
    #
    # `uv run recsys_serving.py --training-run <training-run-id>`
    #
    # Call the endpoint:
    #
    # `curl -X POST https://<app-url>/embed-text "Content-Type: application/json" '{"text": "Testing testing 123"}'`

    app = flyte.with_servecontext(env_vars={"ARTIFACTS_DIR": artifacts_dir.path}).serve(env)
    print(f"Deployed Application: {app.url}")
