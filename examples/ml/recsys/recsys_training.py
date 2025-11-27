# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers[torch]",
#     "torch",
#     "datasets",
#     "scikit-learn",
#     "numpy",
#     "sentence-transformers",
# ]
# ///
"""
Recommendation System - Training Pipeline

This script demonstrates:
1. Creating synthetic user-item interaction data
2. Training a sentence transformer embedding model
3. Generating embeddings for users and items
4. Saving the trained model for serving

The model learns to embed users and items in the same space based on their
interaction history, enabling similarity-based recommendations.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

import flyte
import flyte.io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create image from script dependencies
image = flyte.Image.from_uv_script(__file__, name="recsys_training")

training_env = flyte.TaskEnvironment(
    name="recsys_training",
    image=image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    cache="auto",
)


@training_env.task
async def generate_synthetic_data(
    num_users: int = 1000,
    num_items: int = 500,
    num_interactions: int = 10000,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Generate synthetic user-item interaction data.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        num_interactions: Number of user-item interactions

    Returns:
        Tuple of (users, items, interactions)
    """
    logger.info(f"Generating synthetic data: {num_users} users, {num_items} items, {num_interactions} interactions")

    # Generate users with profiles
    users = []
    categories = ["tech", "sports", "fashion", "food", "travel", "gaming", "music", "books"]

    for user_id in range(num_users):
        # Each user has preferences for certain categories
        preferred_categories = list(np.random.choice(categories, size=np.random.randint(2, 5), replace=False))
        users.append({
            "user_id": f"user_{user_id}",
            "age_group": np.random.choice(["18-25", "26-35", "36-45", "46-60", "60+"]),
            "interests": preferred_categories,
            "profile": f"User interested in {', '.join(preferred_categories)}",
        })

    # Generate items
    items = []
    for item_id in range(num_items):
        category = np.random.choice(categories)
        items.append({
            "item_id": f"item_{item_id}",
            "category": category,
            "title": f"{category.title()} Product {item_id}",
            "description": f"A great {category} item with unique features",
            "tags": list(np.random.choice(["popular", "new", "trending", "sale"], size=np.random.randint(1, 3))),
        })

    # Generate interactions (user views/purchases)
    interactions = []
    for _ in range(num_interactions):
        user = np.random.choice(users)
        # Bias towards items in user's preferred categories
        if np.random.random() < 0.7:  # 70% chance to pick from preferred categories
            matching_items = [item for item in items if item["category"] in user["interests"]]
            if matching_items:
                item = np.random.choice(matching_items)
            else:
                item = np.random.choice(items)
        else:
            item = np.random.choice(items)

        interactions.append({
            "user_id": user["user_id"],
            "item_id": item["item_id"],
            "rating": np.random.randint(3, 6),  # Ratings 3-5 (positive interactions)
            "interaction_type": np.random.choice(["view", "purchase", "like"]),
        })

    logger.info(f"Generated {len(users)} users, {len(items)} items, {len(interactions)} interactions")
    return users, items, interactions


@training_env.task
async def prepare_training_pairs(
    users: List[dict],
    items: List[dict],
    interactions: List[dict],
) -> List[InputExample]:
    """
    Prepare training pairs for contrastive learning.

    Creates positive pairs (user, interacted_item) and uses the interaction
    rating as a similarity score.

    Args:
        users: List of user dictionaries
        items: List of item dictionaries
        interactions: List of interaction dictionaries

    Returns:
        List of InputExample for training
    """
    logger.info("Preparing training pairs...")

    # Create lookup dicts
    user_dict = {u["user_id"]: u for u in users}
    item_dict = {i["item_id"]: i for i in items}

    training_examples = []

    for interaction in interactions:
        user = user_dict[interaction["user_id"]]
        item = item_dict[interaction["item_id"]]

        # Create text representations
        user_text = f"{user['profile']} Age: {user['age_group']}"
        item_text = f"{item['title']}. {item['description']} Category: {item['category']}"

        # Normalize rating to 0-1 range for similarity
        similarity_score = (interaction["rating"] - 3) / 2.0  # 3->0, 5->1

        training_examples.append(
            InputExample(texts=[user_text, item_text], label=similarity_score)
        )

    logger.info(f"Created {len(training_examples)} training pairs")
    return training_examples


@training_env.task
async def train_embedding_model(
    training_examples: List[InputExample],
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_epochs: int = 3,
    batch_size: int = 32,
) -> flyte.io.Dir:
    """
    Train a sentence transformer model on user-item pairs.

    Args:
        training_examples: List of training examples
        base_model: Base model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Directory containing the trained model
    """
    logger.info(f"Training embedding model with {len(training_examples)} examples")

    # Load base model
    model = SentenceTransformer(base_model)

    # Create DataLoader
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)

    # Use CosineSimilarityLoss for learning embeddings
    train_loss = losses.CosineSimilarityLoss(model)

    # Training
    logger.info(f"Starting training for {num_epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=100,
        show_progress_bar=True,
    )

    # Save model
    output_dir = Path("/tmp/recsys_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_dir}")
    model.save(str(output_dir))

    return flyte.io.Dir(str(output_dir))


@training_env.task
async def generate_embeddings(
    model_dir: flyte.io.Dir,
    users: List[dict],
    items: List[dict],
) -> Tuple[dict, dict]:
    """
    Generate embeddings for all users and items using the trained model.

    Args:
        model_dir: Directory containing the trained model
        users: List of user dictionaries
        items: List of item dictionaries

    Returns:
        Tuple of (user_embeddings_dict, item_embeddings_dict)
    """
    logger.info("Loading trained model...")
    model = SentenceTransformer(str(model_dir.path))

    # Generate user embeddings
    logger.info(f"Generating embeddings for {len(users)} users...")
    user_texts = [f"{u['profile']} Age: {u['age_group']}" for u in users]
    user_embeddings = await asyncio.to_thread(model.encode, user_texts, show_progress_bar=True)

    user_embeddings_dict = {
        users[i]["user_id"]: user_embeddings[i].tolist()
        for i in range(len(users))
    }

    # Generate item embeddings
    logger.info(f"Generating embeddings for {len(items)} items...")
    item_texts = [f"{i['title']}. {i['description']} Category: {i['category']}" for i in items]
    item_embeddings = await asyncio.to_thread(model.encode, item_texts, show_progress_bar=True)

    item_embeddings_dict = {
        items[i]["item_id"]: item_embeddings[i].tolist()
        for i in range(len(items))
    }

    logger.info("Embedding generation complete")
    return user_embeddings_dict, item_embeddings_dict


@training_env.task
async def save_artifacts(
    model_dir: flyte.io.Dir,
    user_embeddings: dict,
    item_embeddings: dict,
    users: List[dict],
    items: List[dict],
) -> flyte.io.Dir:
    """
    Save all artifacts needed for serving.

    Args:
        model_dir: Directory containing the trained model
        user_embeddings: User embeddings dictionary
        item_embeddings: Item embeddings dictionary
        users: List of user dictionaries
        items: List of item dictionaries

    Returns:
        Directory containing all artifacts
    """
    output_dir = Path("/tmp/recsys_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    logger.info("Saving embeddings...")
    with open(output_dir / "user_embeddings.json", "w") as f:
        json.dump(user_embeddings, f)

    with open(output_dir / "item_embeddings.json", "w") as f:
        json.dump(item_embeddings, f)

    # Save metadata
    logger.info("Saving metadata...")
    with open(output_dir / "users.json", "w") as f:
        json.dump(users, f)

    with open(output_dir / "items.json", "w") as f:
        json.dump(items, f)

    # Copy model
    import shutil
    model_output = output_dir / "model"
    shutil.copytree(model_dir.path, model_output, dirs_exist_ok=True)

    logger.info(f"All artifacts saved to {output_dir}")
    return flyte.io.Dir(str(output_dir))


@training_env.task
async def training_pipeline(
    num_users: int = 1000,
    num_items: int = 500,
    num_interactions: int = 10000,
    num_epochs: int = 3,
    batch_size: int = 32,
) -> flyte.io.Dir:
    """
    Complete training pipeline for the recommendation system.

    Args:
        num_users: Number of synthetic users
        num_items: Number of synthetic items
        num_interactions: Number of synthetic interactions
        num_epochs: Training epochs
        batch_size: Training batch size

    Returns:
        Directory containing all trained artifacts
    """
    logger.info("=== Starting Recommendation System Training Pipeline ===")

    # Generate data
    users, items, interactions = await generate_synthetic_data(
        num_users=num_users,
        num_items=num_items,
        num_interactions=num_interactions,
    )

    # Prepare training pairs
    training_examples = await prepare_training_pairs(users, items, interactions)

    # Train model
    model_dir = await train_embedding_model(
        training_examples=training_examples,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Generate embeddings
    user_embeddings, item_embeddings = await generate_embeddings(
        model_dir=model_dir,
        users=users,
        items=items,
    )

    # Save all artifacts
    artifacts_dir = await save_artifacts(
        model_dir=model_dir,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        users=users,
        items=items,
    )

    logger.info("=== Training Pipeline Complete ===")
    return artifacts_dir


if __name__ == "__main__":
    flyte.init_from_config()

    # Run the training pipeline
    run = flyte.run(
        training_pipeline,
        num_users=1000,
        num_items=500,
        num_interactions=10000,
        num_epochs=3,
        batch_size=32,
    )
    print(f"Training Run URL: {run.url}")
    run.wait()
    print("Training completed!")
