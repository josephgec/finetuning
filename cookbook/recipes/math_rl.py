"""Math RL recipe: train a model on GSM8K-style problems with verifier rewards.

Usage:
    python -m cookbook.recipes.math_rl \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --steps 50
"""

from __future__ import annotations

import argparse
import re

from local_tinker import (
    AdamParams,
    LoraConfig,
    SamplingParams,
    ServiceClient,
)
from local_tinker.losses import PPOLoss

from cookbook.rl import ProblemEnv, rollout, compute_advantages


SAMPLE_PROBLEMS = [
    {"question": "What is 2 + 3?", "answer": "5"},
    {"question": "What is 7 * 8?", "answer": "56"},
    {"question": "What is 100 - 37?", "answer": "63"},
    {"question": "What is 144 / 12?", "answer": "12"},
    {"question": "What is 15 + 27?", "answer": "42"},
]


def extract_number(text: str) -> str:
    """Extract the last number from a response."""
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Math RL training")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rollouts-per-step", type=int, default=4)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    client = ServiceClient()
    run = client.create_training_run(
        model=args.model,
        lora_config=LoraConfig(rank=16, alpha=32),
        quantize=args.quantize,
    )
    tc = run.training_client()
    sc = run.sampling_client()

    env = ProblemEnv(SAMPLE_PROBLEMS, extract_answer_fn=extract_number)
    sample_params = SamplingParams(max_tokens=64, temperature=0.8)
    loss_fn = PPOLoss(clip_range=0.2)

    for step in range(args.steps):
        # Collect rollouts
        trajectories = [
            rollout(env, sc, run.tokenizer, sample_params)
            for _ in range(args.rollouts_per_step)
        ]

        avg_reward = sum(t.total_reward for t in trajectories) / len(trajectories)

        # Compute advantages and train
        data = compute_advantages(trajectories, run.tokenizer, method="grpo")
        if data:
            result = tc.forward_backward(data, loss_fn=CrossEntropyLoss())
            tc.optim_step(AdamParams(lr=args.lr))
            print(
                f"Step {step + 1}/{args.steps} | "
                f"Loss: {result.loss:.4f} | "
                f"Avg Reward: {avg_reward:.2f}"
            )

    tc.save_weights("./math_rl_output")
    print("Done! Weights saved to ./math_rl_output")


# Need this import for the training loop
from local_tinker.losses import CrossEntropyLoss

if __name__ == "__main__":
    main()
