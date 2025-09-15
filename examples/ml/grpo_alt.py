"""
Simple GRPO (Group Relative Policy Optimization) implementation for fine-tuning Qwen models.
Functional approach with async Python for efficient batch processing and training.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.asyncio import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""

    model_name: str = "Qwen/Qwen2.5-0.5B"  # Small Qwen model
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


class PreferenceDataset(Dataset):
    """Dataset for preference learning with GRPO"""

    def __init__(self, prompts: List[str], tokenizer, max_length: int = 512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


async def generate_single_response(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, config: GRPOConfig
) -> Dict:
    """Generate a single response asynchronously"""
    await asyncio.sleep(0)  # Yield control for async execution

    device = torch.device(config.device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=config.max_length).to(
        device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_length,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": response_text, "token_ids": outputs[0].cpu().numpy(), "prompt": prompt}


async def generate_response_group(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, config: GRPOConfig
) -> List[Dict]:
    """Generate multiple responses for a single prompt"""
    tasks = []
    for _ in range(config.group_size):
        task = generate_single_response(model, tokenizer, prompt, config)
        tasks.append(task)

    responses = await asyncio.gather(*tasks)
    return responses


def compute_rewards(responses: List[Dict], reward_model=None) -> torch.Tensor:
    """
    Compute rewards for generated responses.
    In practice, use a trained reward model or human feedback.
    """
    # Placeholder reward computation
    # Replace this with actual reward model inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if reward_model is not None:
        # Use actual reward model
        # rewards = reward_model(responses)
        pass
    else:
        # Random rewards for demonstration
        rewards = torch.randn(len(responses), device=device)

    return rewards


def compute_log_probs(
    model: AutoModelForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_ref_model: bool = False
) -> torch.Tensor:
    """Compute log probabilities for given sequences"""
    with torch.no_grad() if is_ref_model else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits

    # Shift for autoregressive calculation
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Calculate log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return gathered_log_probs


def compute_grpo_loss(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts_batch: List[str],
    responses_batch: List[List[Dict]],
    config: GRPOConfig,
) -> torch.Tensor:
    """Compute GRPO loss for a batch of prompts and their responses"""
    device = torch.device(config.device)
    total_loss = 0

    for prompt, responses in zip(prompts_batch, responses_batch):
        # Get rewards for all responses
        rewards = compute_rewards(responses)

        # Normalize rewards within the group
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Tokenize all responses
        response_texts = [r["text"] for r in responses]
        tokenized = tokenizer(
            response_texts, return_tensors="pt", padding=True, truncation=True, max_length=config.max_length
        ).to(device)

        # Compute log probabilities for policy and reference models
        policy_log_probs = compute_log_probs(model, tokenized.input_ids, tokenized.attention_mask, is_ref_model=False)

        ref_log_probs = compute_log_probs(ref_model, tokenized.input_ids, tokenized.attention_mask, is_ref_model=True)

        # Compute KL divergence
        kl_div = (policy_log_probs - ref_log_probs).sum(dim=1)

        # GRPO objective: maximize reward while minimizing KL divergence
        advantages = rewards - config.beta * kl_div

        # Compute policy gradient loss
        loss = -(policy_log_probs.sum(dim=1) * advantages).mean()

        total_loss += loss

    return total_loss / len(prompts_batch)


async def train_step(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    batch: List[str],
    config: GRPOConfig,
) -> float:
    """Execute a single training step with async response generation"""
    model.train()

    # Generate multiple responses for each prompt
    all_responses = []
    for prompt in batch:
        responses = await generate_response_group(model, tokenizer, prompt, config)
        all_responses.append(responses)

    # Compute GRPO loss
    loss = compute_grpo_loss(model, ref_model, tokenizer, batch, all_responses, config)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


async def train(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: PreferenceDataset,
    config: GRPOConfig,
) -> AutoModelForCausalLM:
    """Main training loop"""
    device = torch.device(config.device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    logger.info("Starting GRPO training...")

    for epoch in range(config.num_epochs):
        epoch_losses = []

        # Progress bar for async training
        pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            loss = await train_step(model, ref_model, tokenizer, optimizer, batch, config)
            epoch_losses.append(loss)

            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss:.4f}")

        pbar.close()

        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    return model


def save_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path: str):
    """Save the trained model and tokenizer"""
    logger.info(f"Saving model to {path}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_models(config: GRPOConfig) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load the policy model, reference model, and tokenizer"""
    device = torch.device(config.device)
    dtype = torch.float16 if config.device == "cuda" else torch.float32

    logger.info(f"Loading models: {config.model_name}")

    # Load policy model (will be trained)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=dtype, device_map="auto" if config.device == "cuda" else None
    )

    # Load reference model (stays frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=dtype, device_map="auto" if config.device == "cuda" else None
    )
    ref_model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, ref_model, tokenizer


async def evaluate_model(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, test_prompts: List[str], config: GRPOConfig
) -> Dict[str, float]:
    """Evaluate the model on test prompts"""
    model.eval()

    logger.info("Evaluating model...")
    responses = []

    for prompt in test_prompts:
        response = await generate_single_response(model, tokenizer, prompt, config)
        responses.append(response)

    # Compute metrics (placeholder - add your own metrics)
    rewards = compute_rewards(responses)
    avg_reward = rewards.mean().item()

    metrics = {"average_reward": avg_reward, "num_samples": len(responses)}

    return metrics


async def main():
    """Main execution function"""
    # Configuration
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B",  # Using small Qwen model
        batch_size=2,
        num_epochs=3,
        group_size=4,
        learning_rate=1e-5,
        beta=0.1,
    )

    # Sample training data (replace with your actual dataset)
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

    test_prompts = [
        "What is deep learning?",
        "Explain cloud computing:",
    ]

    # Load models and tokenizer
    model, ref_model, tokenizer = load_models(config)

    # Create dataset
    dataset = PreferenceDataset(train_prompts, tokenizer, config.max_length)

    # Train the model
    trained_model = await train(model, ref_model, tokenizer, dataset, config)

    # Evaluate the model
    metrics = await evaluate_model(trained_model, tokenizer, test_prompts, config)
    logger.info(f"Evaluation metrics: {metrics}")

    # Save the fine-tuned model
    save_model(trained_model, tokenizer, "./qwen_grpo_finetuned")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
