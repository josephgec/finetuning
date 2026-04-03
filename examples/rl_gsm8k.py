"""RL training on GSM8K math problems.

Usage:
    python examples/rl_gsm8k.py
"""

from __future__ import annotations

import re

from local_tinker import AdamParams, LoraConfig, SamplingParams, ServiceClient
from local_tinker.losses import CrossEntropyLoss

from cookbook.rl import ProblemEnv, compute_advantages, rollout


PROBLEMS = [
    {"question": "Janet has 3 apples. She buys 5 more. How many does she have?", "answer": "8"},
    {"question": "A store has 20 shirts. 7 are sold. How many are left?", "answer": "13"},
    {"question": "If you multiply 6 by 9, what do you get?", "answer": "54"},
    {"question": "There are 15 birds on a wire. 4 fly away. How many remain?", "answer": "11"},
]


def extract_number(text: str) -> str:
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else ""


def main() -> None:
    client = ServiceClient()
    run = client.create_training_run(
        model="meta-llama/Llama-3.2-1B-Instruct",
        lora_config=LoraConfig(rank=16, alpha=32),
    )
    tc = run.training_client()
    sc = run.sampling_client()

    env = ProblemEnv(PROBLEMS, extract_answer_fn=extract_number)
    sample_params = SamplingParams(max_tokens=64, temperature=0.8)
    loss_fn = CrossEntropyLoss()

    for step in range(20):
        trajectories = [rollout(env, sc, run.tokenizer, sample_params) for _ in range(4)]
        avg_reward = sum(t.total_reward for t in trajectories) / len(trajectories)

        data = compute_advantages(trajectories, run.tokenizer)
        if data:
            result = tc.forward_backward(data, loss_fn)
            tc.optim_step(AdamParams(lr=1e-5))
            print(f"Step {step + 1} | Loss: {result.loss:.4f} | Reward: {avg_reward:.2f}")

    tc.save_weights("./rl_gsm8k_output")
    print("Done!")


if __name__ == "__main__":
    main()
