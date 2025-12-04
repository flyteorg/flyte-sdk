# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flyte>=2.0.0b34",
#     "transformers[torch]",
#     "torch",
#     "datasets",
#     "scikit-learn",
#     "numpy",
#     "pandas",
#     "pyarrow",
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
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
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
    cache=flyte.Cache("auto", "1.0"),
)


@training_env.task
async def generate_synthetic_data(
    num_users: int = 1000,
    num_items: int = 500,
    num_interactions: int = 10000,
) -> Tuple[flyte.io.DataFrame, flyte.io.DataFrame, flyte.io.DataFrame]:
    """
    Generate synthetic user-item interaction data as flyte.io.DataFrame references.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        num_interactions: Number of user-item interactions

    Returns:
        Tuple of (users_df, items_df, interactions_df) as flyte.io.DataFrame references
    """
    logger.info(f"Generating synthetic data: {num_users} users, {num_items} items, {num_interactions} interactions")

    categories = ["tech", "sports", "fashion", "food", "travel", "gaming", "music", "books"]

    # Generate users DataFrame
    user_data = {
        "user_id": [f"user_{i}" for i in range(num_users)],
        "age_group": np.random.choice(["18-25", "26-35", "36-45", "46-60", "60+"], size=num_users),
        "interests": [
            ",".join(np.random.choice(categories, size=np.random.randint(2, 5), replace=False))
            for _ in range(num_users)
        ],
    }
    users_df = pd.DataFrame(user_data)
    users_df["profile"] = users_df["interests"].apply(lambda x: f"User interested in {x.replace(',', ', ')}")

    # Generate items DataFrame
    item_data = {
        "item_id": [f"item_{i}" for i in range(num_items)],
        "category": np.random.choice(categories, size=num_items),
        "title": [f"{np.random.choice(categories).title()} Product {i}" for i in range(num_items)],
        "description": [f"A great {np.random.choice(categories)} item with unique features" for _ in range(num_items)],
        "tags": [
            ",".join(np.random.choice(["popular", "new", "trending", "sale"], size=np.random.randint(1, 3)))
            for _ in range(num_items)
        ],
    }
    items_df = pd.DataFrame(item_data)

    # Generate interactions DataFrame
    interaction_data = []
    for _ in range(num_interactions):
        # Sample random user
        user_idx = np.random.randint(0, num_users)
        user_interests = users_df.iloc[user_idx]["interests"].split(",")

        # Bias towards items in user's preferred categories
        if np.random.random() < 0.7:  # 70% chance to pick from preferred categories
            matching_items = items_df[items_df["category"].isin(user_interests)]
            if len(matching_items) > 0:
                item_idx = matching_items.sample(1).index[0]
            else:
                item_idx = np.random.randint(0, num_items)
        else:
            item_idx = np.random.randint(0, num_items)

        interaction_data.append(
            {
                "user_id": users_df.iloc[user_idx]["user_id"],
                "item_id": items_df.iloc[item_idx]["item_id"],
                "rating": np.random.randint(3, 6),  # Ratings 3-5 (positive interactions)
                "interaction_type": np.random.choice(["view", "purchase", "like"]),
            }
        )

    interactions_df = pd.DataFrame(interaction_data)

    logger.info(f"Generated {len(users_df)} users, {len(items_df)} items, {len(interactions_df)} interactions")

    # Convert to flyte.io.DataFrame to avoid materialization when passing between tasks
    return (
        flyte.io.DataFrame.from_df(users_df),
        flyte.io.DataFrame.from_df(items_df),
        flyte.io.DataFrame.from_df(interactions_df),
    )


@training_env.task
async def prepare_training_pairs(
    users_df: flyte.io.DataFrame,
    items_df: flyte.io.DataFrame,
    interactions_df: flyte.io.DataFrame,
) -> List[InputExample]:
    """
    Prepare training pairs for contrastive learning.

    Creates positive pairs (user, interacted_item) and uses the interaction
    rating as a similarity score.

    Args:
        users_df: Users DataFrame reference with columns [user_id, age_group, interests, profile]
        items_df: Items DataFrame reference with columns [item_id, category, title, description, tags]
        interactions_df: Interactions DataFrame reference with columns [user_id, item_id, rating, interaction_type]

    Returns:
        List of InputExample for training
    """
    logger.info("Preparing training pairs...")

    # Materialize the DataFrames only when needed for computation
    users_pd = await users_df.open(pd.DataFrame).all()
    items_pd = await items_df.open(pd.DataFrame).all()
    interactions_pd = await interactions_df.open(pd.DataFrame).all()

    # Merge interactions with user and item data
    merged_df = interactions_pd.merge(users_pd, on="user_id").merge(items_pd, on="item_id")

    training_examples = []
    for _, row in merged_df.iterrows():
        # Create text representations
        user_text = f"{row['profile']} Age: {row['age_group']}"
        item_text = f"{row['title']}. {row['description']} Category: {row['category']}"

        # Normalize rating to 0-1 range for similarity
        similarity_score = (row["rating"] - 3) / 2.0  # 3->0, 5->1

        training_examples.append(InputExample(texts=[user_text, item_text], label=similarity_score))

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

    return await flyte.io.Dir.from_local(output_dir)


@training_env.task
async def generate_embeddings(
    model_dir: flyte.io.Dir,
    users_df: flyte.io.DataFrame,
    items_df: flyte.io.DataFrame,
) -> Tuple[flyte.io.DataFrame, flyte.io.DataFrame]:
    """
    Generate embeddings for all users and items using the trained model.

    Args:
        model_dir: Directory containing the trained model
        users_df: Users DataFrame reference
        items_df: Items DataFrame reference

    Returns:
        Tuple of (user_embeddings_df, item_embeddings_df) as flyte.io.DataFrame references
    """
    logger.info("Loading trained model...")
    path = await model_dir.download()
    model = SentenceTransformer(path)

    # Materialize DataFrames for processing
    users_pd = await users_df.open(pd.DataFrame).all()
    items_pd = await items_df.open(pd.DataFrame).all()

    # Generate user embeddings
    logger.info(f"Generating embeddings for {len(users_pd)} users...")
    user_texts = [f"{row['profile']} Age: {row['age_group']}" for _, row in users_pd.iterrows()]
    user_embeddings = await asyncio.to_thread(model.encode, user_texts, show_progress_bar=True)

    user_embeddings_df = users_pd.copy()
    user_embeddings_df["embedding"] = list(user_embeddings)

    # Generate item embeddings
    logger.info(f"Generating embeddings for {len(items_pd)} items...")
    item_texts = [f"{row['title']}. {row['description']} Category: {row['category']}" for _, row in items_pd.iterrows()]
    item_embeddings = await asyncio.to_thread(model.encode, item_texts, show_progress_bar=True)

    item_embeddings_df = items_pd.copy()
    item_embeddings_df["embedding"] = list(item_embeddings)

    logger.info("Embedding generation complete")

    # Convert to flyte.io.DataFrame to avoid materialization when passing between tasks
    return (
        flyte.io.DataFrame.from_df(user_embeddings_df),
        flyte.io.DataFrame.from_df(item_embeddings_df),
    )


@training_env.task
async def save_artifacts(
    model_dir: flyte.io.Dir,
    user_embeddings_df: flyte.io.DataFrame,
    item_embeddings_df: flyte.io.DataFrame,
) -> flyte.io.Dir:
    """
    Save all artifacts needed for serving.

    Args:
        model_dir: Directory containing the trained model
        user_embeddings_df: DataFrame reference with user data and embeddings
        item_embeddings_df: DataFrame reference with item data and embeddings

    Returns:
        Directory containing all artifacts
    """
    output_dir = Path("/tmp/recsys_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Materialize DataFrames for saving
    user_embeddings_pd = await user_embeddings_df.open(pd.DataFrame).all()
    item_embeddings_pd = await item_embeddings_df.open(pd.DataFrame).all()

    # Save DataFrames as parquet (efficient columnar format)
    logger.info("Saving embeddings and metadata as parquet...")
    user_embeddings_pd.to_parquet(output_dir / "user_embeddings.parquet", index=False)
    item_embeddings_pd.to_parquet(output_dir / "item_embeddings.parquet", index=False)

    # Copy model
    import shutil

    model_output = output_dir / "model"
    path = await model_dir.download()
    shutil.copytree(path, model_output, dirs_exist_ok=True)

    logger.info(f"All artifacts saved to {output_dir}")
    return await flyte.io.Dir.from_local(output_dir)


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
        users_df=users,
        items_df=items,
    )

    # Save all artifacts
    artifacts_dir = await save_artifacts(
        model_dir=model_dir,
        user_embeddings_df=user_embeddings,
        item_embeddings_df=item_embeddings,
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
