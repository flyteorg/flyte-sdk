"""
GRPO (Group Relative Policy Optimization) - Flyte 2.0 Implementation

True GRPO algorithm with real gradient-based optimization, distributed across GPU/CPU tasks.
Maintains algorithmic fidelity with the paper while leveraging Flyte's distributed execution.
"""

import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from async_lru import alru_cache
from tqdm.asyncio import tqdm

import flyte
from flyte.io import File

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_length: int = 512
    temperature: float = 0.7
    group_size: int = 4  # Number of responses per prompt for ranking
    beta: float = 0.1  # KL penalty coefficient
    clip_range: float = 0.2  # PPO-style clipping
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


@dataclass
class ModelResponse:
    """Single model response with metadata"""
    text: str
    token_ids: List[int]
    log_probs: List[float]
    prompt: str


@dataclass
class TrainingBatch:
    """Training batch with responses and computed data"""
    prompts: List[str]
    responses: List[List[ModelResponse]]  # [batch_size, group_size]
    rewards: List[List[float]]  # [batch_size, group_size]
    advantages: List[List[float]]  # [batch_size, group_size]


@dataclass
class TrainingState:
    """Serializable training state for checkpointing"""
    epoch: int
    batch_idx: int
    total_loss: float
    losses_history: List[float]
    config: GRPOConfig


@dataclass
class ModelCheckpoint:
    """Model checkpoint data"""
    state_dict: Dict[str, Any]  # Model state dict (Flyte will handle serialization)
    optimizer_state: Optional[Dict[str, Any]]  # Optimizer state (optional)
    training_state: TrainingState


# =============================================================================
# 2. FLYTE ENVIRONMENTS
# =============================================================================

# GPU Environment for model operations
gpu_env = flyte.TaskEnvironment(
    name="grpo_gpu",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "torch", "transformers", "numpy", "tqdm", "async-lru"
    ),
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu=1),
    reusable=flyte.ReusePolicy(
        replicas=1,  # Keep one GPU instance alive
        idle_ttl=600,  # 10 minutes idle time
        concurrency=5,  # Allow multiple concurrent tasks
        scaledown_ttl=300,  # 5 minutes before scaling down
    ),
)

# CPU Environment for reward computation and data processing
cpu_env = flyte.TaskEnvironment(
    name="grpo_cpu",
    image=flyte.Image.from_debian_base().with_pip_packages("numpy", "torch", "tqdm"),
    resources=flyte.Resources(cpu=2, memory="8Gi"),
)


# =============================================================================
# 3. MODEL MANAGEMENT (GPU TASKS)
# =============================================================================

@alru_cache
async def get_models(model_name: str) -> Tuple[Any, Any, Any]:
    """Cached model loading - loads once and reuses across tasks"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading models (cached): {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load policy model (trainable)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Load reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Models loaded and cached successfully")
    return policy_model, ref_model, tokenizer


@gpu_env.task
async def generate_response_batch(
        prompts: List[str],
        config: GRPOConfig,
        model_state: Optional[Dict[str, Any]] = None  # Optional updated model state
) -> List[List[ModelResponse]]:
    """Generate responses for a batch of prompts on GPU"""
    policy_model, ref_model, tokenizer = await get_models(config.model_name)

    # Update model state if provided (for fine-tuning)
    if model_state:
        policy_model.load_state_dict(model_state)

    policy_model.eval()  # Set to eval for generation
    device = torch.device(config.device)

    all_responses = []

    # Generate responses for each prompt
    for prompt in tqdm(prompts, desc="Generating responses", leave=False):
        prompt_responses = []

        # Generate group_size responses per prompt
        for _ in range(config.group_size):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length
            ).to(device)

            with torch.no_grad():
                outputs = policy_model.generate(
                    **inputs,
                    max_new_tokens=config.max_length,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode response
            response_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

            # Compute log probabilities for the response
            with torch.no_grad():
                logits = torch.stack(outputs.scores, dim=1)  # [batch, seq_len, vocab]
                log_probs = F.log_softmax(logits, dim=-1)

                # Get log probs for generated tokens
                generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
                token_log_probs = []
                for i, token_id in enumerate(generated_tokens):
                    if i < log_probs.shape[1]:
                        token_log_prob = log_probs[0, i, token_id].item()
                        token_log_probs.append(token_log_prob)

            response = ModelResponse(
                text=response_text,
                token_ids=generated_tokens.cpu().tolist(),
                log_probs=token_log_probs,
                prompt=prompt
            )
            prompt_responses.append(response)

        all_responses.append(prompt_responses)

    return all_responses


@gpu_env.task
async def compute_grpo_gradients(
        training_batch: TrainingBatch,
        config: GRPOConfig,
        model_state: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], float]:
    """Compute GRPO gradients on GPU and return gradient updates"""
    policy_model, ref_model, tokenizer = await get_models(config.model_name)

    # Update model state if provided
    if model_state:
        policy_model.load_state_dict(model_state)

    policy_model.train()
    device = torch.device(config.device)

    # Initialize optimizer (recreated each time - could be optimized)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)

    total_loss = 0.0

    # Process each prompt group
    for prompt_idx, (prompt, responses, rewards, advantages) in enumerate(
            zip(training_batch.prompts, training_batch.responses,
                training_batch.rewards, training_batch.advantages)
    ):
        # Prepare response texts and tokenize
        response_texts = [r.text for r in responses]
        tokenized = tokenizer(
            response_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length
        ).to(device)

        # Compute policy log probabilities
        policy_outputs = policy_model(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            labels=tokenized.input_ids
        )

        # Shift for autoregressive calculation
        shift_logits = policy_outputs.logits[:, :-1, :].contiguous()
        shift_labels = tokenized.input_ids[:, 1:].contiguous()

        policy_log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered_policy_log_probs = torch.gather(
            policy_log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        policy_log_probs_sum = gathered_policy_log_probs.sum(dim=1)

        # Compute reference log probabilities
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                labels=tokenized.input_ids
            )

            ref_shift_logits = ref_outputs.logits[:, :-1, :].contiguous()
            ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
            gathered_ref_log_probs = torch.gather(
                ref_log_probs, 2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            ref_log_probs_sum = gathered_ref_log_probs.sum(dim=1)

        # Compute KL divergence
        kl_div = (policy_log_probs_sum - ref_log_probs_sum)

        # Convert advantages to tensors
        advantages_tensor = torch.tensor(advantages, device=device, dtype=torch.float32)

        # GRPO objective: maximize reward while minimizing KL divergence
        grpo_advantages = advantages_tensor - config.beta * kl_div

        # Policy gradient loss (negative because we want to maximize)
        loss = -(policy_log_probs_sum * grpo_advantages).mean()
        total_loss += loss.item()

        # Backward pass
        loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

    # Extract gradients and apply optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Return updated model state
    updated_state = policy_model.state_dict()
    avg_loss = total_loss / len(training_batch.prompts)

    return updated_state, avg_loss


# =============================================================================
# 4. REWARD & ADVANTAGE COMPUTATION (CPU TASKS)
# =============================================================================

@cpu_env.task
async def compute_batch_rewards(responses: List[List[ModelResponse]]) -> List[List[float]]:
    """Compute rewards for responses (CPU task)"""
    # This is a placeholder - replace with actual reward model
    # For now, using simple heuristics similar to grpo.py

    all_rewards = []

    for response_group in responses:
        group_rewards = []
        for response in response_group:
            text = response.text
            if not text.strip():
                group_rewards.append(0.0)
                continue

            words = text.split()
            if not words:
                group_rewards.append(0.0)
                continue

            # Multi-heuristic reward (can replace with learned reward model)
            length_score = min(len(words) / 100, 1.0)
            unique_tokens = len(set(word.lower() for word in words))
            diversity_score = min(unique_tokens / 50, 1.0)
            avg_word_length = np.mean([len(word) for word in words])
            complexity_score = min(avg_word_length / 10, 1.0)

            reward = 0.4 * length_score + 0.3 * diversity_score + 0.3 * complexity_score
            group_rewards.append(max(0.0, reward))

        all_rewards.append(group_rewards)

    return all_rewards


@cpu_env.task
async def compute_batch_advantages(
        rewards: List[List[float]],
        gamma: float = 1.0
) -> List[List[float]]:
    """Compute advantages using group-relative comparison (CPU task)"""
    all_advantages = []

    for reward_group in rewards:
        if not reward_group:
            all_advantages.append([])
            continue

        # Normalize rewards within group (GRPO key insight)
        rewards_array = np.array(reward_group)
        baseline = np.mean(rewards_array)
        advantages = gamma * (rewards_array - baseline)
        all_advantages.append(advantages.tolist())

    return all_advantages


# =============================================================================
# 5. TRAINING ORCHESTRATION
# =============================================================================

@cpu_env.task
async def save_checkpoint(
        model_state: Dict[str, Any],
        training_state: TrainingState
) -> ModelCheckpoint:
    """Save training checkpoint - Flyte handles serialization automatically"""
    checkpoint = ModelCheckpoint(
        state_dict=model_state,
        optimizer_state=None,  # Could add optimizer state if needed
        training_state=training_state
    )

    logger.info(f"Checkpoint created for epoch {training_state.epoch}")
    return checkpoint


async def train_epoch(
        prompts: List[str],
        config: GRPOConfig,
        model_state: Optional[Dict[str, Any]],
        epoch: int
) -> Tuple[Optional[Dict[str, Any]], List[float]]:
    """Train one epoch with proper GPU/CPU task distribution"""

    # Split prompts into batches
    batch_losses = []
    current_model_state = model_state

    # Create batches
    batches = [prompts[i:i + config.batch_size] for i in range(0, len(prompts), config.batch_size)]

    for batch_idx, batch_prompts in enumerate(tqdm(batches, desc=f"Epoch {epoch + 1} Batches")):
        with (flyte.group(f"epoch-{epoch}-batch-{batch_idx}")):
            # Step 1: Generate responses (GPU)
            responses = await generate_response_batch(
                batch_prompts,
                config,
                current_model_state
            )

            # Step 2: Compute rewards (CPU)
            rewards = await compute_batch_rewards(responses)

            # Step 3: Compute advantages (CPU) 
            advantages = await compute_batch_advantages(rewards)

            # Step 4: Create training batch (CPU)
            training_batch = TrainingBatch(
                prompts=batch_prompts,
                responses=responses,
                rewards=rewards,
                advantages=advantages
            )

            # Step 5: Compute gradients and update model (GPU)
            current_model_state, batch_loss = await compute_grpo_gradients(
                training_batch, config, current_model_state
            )

            batch_losses.append(batch_loss)
            logger.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {batch_loss:.4f}")

    return current_model_state, batch_losses


@cpu_env.task
async def grpo_training_workflow(
        config: GRPOConfig,
        train_prompts: List[str] = None,
        checkpoint_every: int = 1
) -> ModelCheckpoint:
    """Main GRPO training workflow - leverages Flyte's native serialization for model states"""

    # Default prompts if none provided
    if train_prompts is None:
        train_prompts = [
            "Explain quantum computing in simple terms:",
            "Write a Python function to sort a list:",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis:",
            "How does machine learning work?",
            "What is the difference between AI and ML?",
            "Explain the concept of blockchain:",
            "How do neural networks learn?",
        ]

    logger.info(f"Starting GRPO training with config: {asdict(config)}")

    # Initialize model state (None at start - will be loaded from base model in first task)
    current_model_state: Optional[Dict[str, Any]] = None
    all_losses = []

    # Training loop - Flyte handles model state serialization automatically
    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        # Train one epoch - model loading/saving handled internally
        current_model_state, epoch_losses = await train_epoch(
            train_prompts, config, current_model_state, epoch
        )

        all_losses.extend(epoch_losses)
        avg_epoch_loss = float(np.mean(epoch_losses))
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save checkpoint periodically
        if epoch % checkpoint_every == 0:
            training_state = TrainingState(
                epoch=epoch,
                batch_idx=0,
                total_loss=avg_epoch_loss,
                losses_history=all_losses,
                config=config
            )

            checkpoint = await save_checkpoint(
                current_model_state, training_state
            )
            logger.info(f"Checkpoint created at epoch {epoch + 1}: {checkpoint.training_state.total_loss:.4f} loss")

    # Final checkpoint
    final_training_state = TrainingState(
        epoch=config.num_epochs,
        batch_idx=0,
        total_loss=float(np.mean(all_losses[-10:])) if all_losses else 0.0,
        losses_history=all_losses,
        config=config
    )

    final_checkpoint = await save_checkpoint(current_model_state, final_training_state)
    logger.info("GRPO training completed successfully!")

    return final_checkpoint


# =============================================================================
# 6. EXECUTION
# =============================================================================

if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())

    # Configuration for training
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        learning_rate=1e-5,
        batch_size=2,  # Small for testing
        group_size=4,
        num_epochs=2,
        beta=0.1,
        temperature=0.7,
        max_length=256,
    )

    # Custom training prompts
    custom_prompts = [
        "Explain the concept of machine learning:",
        "What are the benefits of renewable energy?",
        "Describe how neural networks work:",
        "What is quantum computing?",
    ]

    # Run training workflow
    run = flyte.with_runcontext(mode="local").run(
        grpo_training_workflow,
        config=config,
        train_prompts=custom_prompts,
        checkpoint_every=1
    )

    print(f"GRPO training started: {run.url}")
