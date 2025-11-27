# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "sentence-transformers",
#     "numpy",
#     "pydantic",
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
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Recommendation System API",
    description="Embedding-based recommendation service",
    version="1.0.0",
)

# Global state
model: Optional[SentenceTransformer] = None
user_embeddings: dict = {}
item_embeddings: dict = {}
users_metadata: dict = {}
items_metadata: dict = {}


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
    Load the trained model and pre-computed embeddings.

    Args:
        artifacts_dir: Path to directory containing all artifacts
    """
    global model, user_embeddings, item_embeddings, users_metadata, items_metadata

    logger.info("Loading model...")
    model_path = artifacts_dir / "model"
    model = await asyncio.to_thread(SentenceTransformer, str(model_path))

    logger.info("Loading user embeddings...")
    with open(artifacts_dir / "user_embeddings.json", "r") as f:
        user_embeddings = json.load(f)

    logger.info("Loading item embeddings...")
    with open(artifacts_dir / "item_embeddings.json", "r") as f:
        item_embeddings = json.load(f)

    logger.info("Loading user metadata...")
    with open(artifacts_dir / "users.json", "r") as f:
        users_list = json.load(f)
        users_metadata = {u["user_id"]: u for u in users_list}

    logger.info("Loading item metadata...")
    with open(artifacts_dir / "items.json", "r") as f:
        items_list = json.load(f)
        items_metadata = {i["item_id"]: i for i in items_list}

    logger.info("Model and data loaded successfully")


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
    exclude_items: List[str] = None,
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
    for item_id, item_embedding in item_embeddings.items():
        if item_id in exclude_set:
            continue

        similarity = await compute_similarity(user_embedding, item_embedding)
        similarities.append((item_id, similarity))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Take top-k and create response
    results = []
    for item_id, score in similarities[:top_k]:
        item = items_metadata[item_id]
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


@app.on_event("startup")
async def startup_event():
    """
    Load model and data on startup.
    """
    # TODO: Replace with actual path to your trained artifacts
    artifacts_dir = Path("/tmp/recsys_artifacts")

    if not artifacts_dir.exists():
        logger.warning(f"Artifacts directory not found at {artifacts_dir}")
        logger.warning("Please run the training pipeline first and update the path")
        return

    await load_model_and_data(artifacts_dir)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "num_users": len(user_embeddings),
        "num_items": len(item_embeddings),
    }


@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    Get personalized item recommendations for a user.

    Returns items ranked by similarity to the user's embedding.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.user_id not in user_embeddings:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")

    logger.info(f"Getting recommendations for user {request.user_id}")

    # Get user embedding
    user_embedding = user_embeddings[request.user_id]

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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.item_id not in item_embeddings:
        raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")

    logger.info(f"Finding similar items for {request.item_id}")

    # Get item embedding
    item_embedding = item_embeddings[request.item_id]

    # Compute similarities to all other items
    similarities = []
    for other_item_id, other_embedding in item_embeddings.items():
        if other_item_id == request.item_id:
            continue

        similarity = await compute_similarity(item_embedding, other_embedding)
        similarities.append((other_item_id, similarity))

    # Sort and take top-k
    similarities.sort(key=lambda x: x[1], reverse=True)

    similar_items = []
    for item_id, score in similarities[:request.top_k]:
        item = items_metadata[item_id]
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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Embedding text: {request.text[:50]}...")

    # Generate embedding
    embedding = await asyncio.to_thread(model.encode, request.text)

    return EmbedTextResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding),
    )


@app.get("/users/{user_id}")
async def get_user_info(user_id: str):
    """
    Get user metadata.
    """
    if user_id not in users_metadata:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    return users_metadata[user_id]


@app.get("/items/{item_id}")
async def get_item_info(item_id: str):
    """
    Get item metadata.
    """
    if item_id not in items_metadata:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    return items_metadata[item_id]


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
