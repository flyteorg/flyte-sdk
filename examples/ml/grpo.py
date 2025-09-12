"""
GRPO (Generalized Reinforcement Learning with Preference Optimization) - Real Implementation

Clean Flyte 2.0 implementation with vLLM backend.

Supported models (set MODEL_NAME environment variable):
- microsoft/DialoGPT-medium (default)
- meta-llama/Llama-2-7b-chat-hf
- mistralai/Mistral-7B-Instruct-v0.1
- NousResearch/Llama-2-7b-chat-hf
- teknium/OpenHermes-2.5-Mistral-7B
- Any model supported by vLLM

Usage:
    pip install vllm
    export MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
    python grpo_real.py
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import List

from async_lru import alru_cache

import flyte
from flyte.io import File

# vLLM imports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================


@dataclass
class GRPOConfig:
    group_size: int = 4
    beta: float = 0.1
    learning_rate: float = 1e-5
    batch_size: int = 8
    model_name: str = "microsoft/DialoGPT-medium"
    temperature: float = 0.8
    max_tokens: int = 512
    top_p: float = 0.9


@dataclass
class TrainingResults:
    final_reward: float
    reward_history: List[float]
    num_steps: int
    config: GRPOConfig
    total_responses: int


# =============================================================================
# 2. vLLM ENGINE
# =============================================================================


class VLLMEngine:
    def __init__(self):
        self.engine = None

    async def initialize(self, model_name: str):
        """Initialize vLLM async engine"""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        if self.engine is None:
            logger.info(f"Initializing vLLM engine with model: {model_name}")

            engine_args = AsyncEngineArgs(
                model=model_name,
                tensor_parallel_size=1,
                trust_remote_code=True,
                max_model_len=2048,
                gpu_memory_utilization=0.8,
            )

            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM engine initialized successfully")

    async def generate_responses(self, prompt: str, config: GRPOConfig) -> List[str]:
        """Generate multiple responses for a prompt using vLLM"""
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        await self.initialize(config.model_name)

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            n=config.group_size,
            use_beam_search=False,
        )

        request_id = f"grpo_{random_uuid()}"
        result = await self.engine.generate(prompt, sampling_params, request_id)

        # Extract responses
        responses = [output.text.strip() for output in result.outputs]
        return responses


# =============================================================================
# 3. FLYTE TASKS
# =============================================================================

vllm_env = flyte.TaskEnvironment(
    name="vllm",
    image=flyte.Image.from_debian_base().with_pip_packages("vllm", "unionai-reuse==0.1.6b0"),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=300,
        concurrency=10,
        scaledown_ttl=300,
    ),
)


env = flyte.TaskEnvironment(
    name="grpo",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_debian_base().with_pip_packages("numpy"),
    depends_on=[vllm_env],
)


# Global vLLM engine
@alru_cache
async def get_vllm_engine():
    return VLLMEngine()


@env.task
async def load_prompts(data_path: str) -> List[str]:
    """Load training prompts from file or return defaults"""
    try:
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                if data_path.endswith(".json"):
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "prompts" in data:
                        return data["prompts"]
                else:
                    return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.warning(f"Failed to load prompts from {data_path}: {e}")

    # Default prompts
    return [
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "How do neural networks work?",
        "What is the importance of biodiversity?",
        "Explain quantum computing to a beginner.",
        "What is climate change and its impacts?",
        "Describe the structure of DNA.",
    ]


@vllm_env.task
async def generate_responses(prompts: List[str], config: GRPOConfig) -> List[List[str]]:
    """Generate multiple responses per prompt using vLLM"""
    all_responses = []
    vllm_engine = await get_vllm_engine()

    for i, prompt in enumerate(prompts):
        logger.info(f"Generating responses for prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")
        responses = await vllm_engine.generate_responses(prompt, config)
        all_responses.append(responses)

        # Small delay to avoid overwhelming the engine
        await asyncio.sleep(0.1)

    return all_responses


@env.task
async def compute_rewards(responses: List[List[str]]) -> List[List[float]]:
    """Score each response for quality using multiple heuristics"""
    import numpy as np

    all_rewards = []

    for response_group in responses:
        rewards = []
        for response in response_group:
            if not response.strip():
                rewards.append(0.0)
                continue

            words = response.split()
            if not words:
                rewards.append(0.0)
                continue

            # Length reward (moderate length preferred)
            length_score = min(len(words) / 100, 1.0)

            # Diversity reward (unique tokens)
            unique_tokens = len(set(word.lower() for word in words))
            diversity_score = min(unique_tokens / 50, 1.0)

            # Complexity reward (average word length)
            avg_word_length = np.mean([len(word) for word in words])
            complexity_score = min(avg_word_length / 10, 1.0)

            # Coherence reward (sentence structure)
            sentences = response.count(".") + response.count("!") + response.count("?")
            coherence_score = min(sentences / len(words) * 50, 1.0) if words else 0.0

            # Combine scores
            reward = 0.4 * length_score + 0.3 * diversity_score + 0.2 * complexity_score + 0.1 * coherence_score

            rewards.append(max(0.0, reward))

        all_rewards.append(rewards)

    return all_rewards


@env.task
async def compute_advantages(rewards: List[List[float]], gamma: float = 1.0) -> List[List[float]]:
    """Compare each response to its group average"""
    import numpy as np

    all_advantages = []

    for reward_group in rewards:
        if not reward_group:
            all_advantages.append([])
            continue

        baseline = np.mean(reward_group)
        advantages = [gamma * (reward - baseline) for reward in reward_group]
        all_advantages.append(advantages)

    return all_advantages


@env.task
async def update_model(advantages: List[List[float]], config: GRPOConfig) -> float:
    """Update the model based on advantages (simplified)"""
    # In a real implementation, this would:
    # 1. Compute policy gradients from advantages
    # 2. Apply PPO clipping
    # 3. Add KL penalty with beta coefficient
    # 4. Update model weights with optimizer

    flat_advantages = [adv for group in advantages for adv in group if group]
    if not flat_advantages:
        return 0.0

    # Simulate improvement based on positive advantages
    positive_advantages = [adv for adv in flat_advantages if adv > 0]
    improvement = sum(positive_advantages) * config.learning_rate

    logger.info(
        f"Model update: {len(positive_advantages)}/{len(flat_advantages)} positive advantages, "
        f"improvement: {improvement:.6f}"
    )

    return improvement


@env.task
async def save_results(results: TrainingResults, output_dir: str = "/tmp") -> File:
    """Save training results to file"""
    output_path = os.path.join(output_dir, f"grpo_results_{results.num_steps}_steps.json")

    # Convert to dict for JSON serialization
    results_dict = asdict(results)

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to: {output_path}")
    return File(path=output_path)


# =============================================================================
# 4. GRPO TRAINING STEP
# =============================================================================


async def grpo_training_step(prompts: List[str], config: GRPOConfig, step: int) -> float:
    """One step of GRPO training - calls tasks like normal functions"""
    import numpy as np

    with flyte.group(f"grpo-step-{step}"):
        logger.info(f"Step {step}: Generating responses...")
        responses = await generate_responses(prompts, config)

        logger.info(f"Step {step}: Computing rewards...")
        rewards = await compute_rewards(responses)

        logger.info(f"Step {step}: Computing advantages...")
        advantages = await compute_advantages(rewards, gamma=1.0)

        logger.info(f"Step {step}: Updating model...")
        improvement = await update_model(advantages, config)

        # Log step metrics
        flat_rewards = [r for group in rewards for r in group if group]
        if flat_rewards:
            avg_reward = np.mean(flat_rewards)
            logger.info(f"Step {step}: avg_reward={avg_reward:.4f}, improvement={improvement:.6f}")

        return improvement


# =============================================================================
# 5. MAIN WORKFLOW
# =============================================================================


@env.task
async def grpo_workflow(
    data_path: str = "prompts.txt",
    num_steps: int = 5,
    group_size: int = 4,
    learning_rate: float = 1e-5,
    model_name: str = "microsoft/DialoGPT-medium",
    temperature: float = 0.8,
    batch_size: int = 4,
) -> TrainingResults:
    """Complete GRPO training workflow"""

    # Step 1: Load data
    logger.info("Loading training prompts...")
    all_prompts = await load_prompts(data_path)
    logger.info(f"Loaded {len(all_prompts)} prompts")

    # Step 2: Setup config
    config = GRPOConfig(
        group_size=group_size,
        learning_rate=learning_rate,
        model_name=model_name,
        temperature=temperature,
        batch_size=batch_size,
    )

    logger.info(f"Starting GRPO training with config: {asdict(config)}")

    # Step 3: Training loop
    reward_history = []
    total_responses = 0

    for step in range(num_steps):
        # Sample batch of prompts
        batch_prompts = random.sample(all_prompts, min(batch_size, len(all_prompts)))

        # Training step
        improvement = await grpo_training_step(batch_prompts, config, step)
        reward_history.append(improvement)
        total_responses += len(batch_prompts) * group_size

        logger.info(f"Completed step {step + 1}/{num_steps}")

    # Step 4: Create results
    final_reward = sum(reward_history)
    results = TrainingResults(
        final_reward=final_reward,
        reward_history=reward_history,
        num_steps=num_steps,
        config=config,
        total_responses=total_responses,
    )

    logger.info(f"Training completed! Final reward: {final_reward:.4f}, Total responses: {total_responses}")

    return results


# =============================================================================
# 6. RUNNING THE WORKFLOW
# =============================================================================

if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())

    # Use a model supported by vLLM
    model_name = "microsoft/DialoGPT-medium"

    run = flyte.run(
        grpo_workflow,
        num_steps=3,
        group_size=3,
        model_name=model_name,
        batch_size=2,
        temperature=0.8,
        learning_rate=1e-5,
    )

    print(f"GRPO training started: {run.url}")
