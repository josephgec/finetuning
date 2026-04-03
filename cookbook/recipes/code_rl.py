"""Code RL recipe: train a model on code problems with execution-based rewards.

Usage:
    python -m cookbook.recipes.code_rl \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --steps 50
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from typing import Any

from local_tinker import (
    AdamParams,
    LoraConfig,
    SamplingParams,
    ServiceClient,
)
from local_tinker.losses import CrossEntropyLoss

from cookbook.rl import Env, rollout, compute_advantages


SAMPLE_PROBLEMS = [
    {
        "prompt": "Write a Python function that returns the sum of two numbers a and b.",
        "test": "assert solution(2, 3) == 5\nassert solution(-1, 1) == 0",
    },
    {
        "prompt": "Write a Python function that returns the factorial of n.",
        "test": "assert solution(5) == 120\nassert solution(0) == 1",
    },
    {
        "prompt": "Write a Python function that reverses a string s.",
        "test": 'assert solution("hello") == "olleh"\nassert solution("") == ""',
    },
]


class CodeEnv(Env):
    """RL environment that rewards correct code via execution.

    Args:
        problems: List of ``{"prompt": str, "test": str}`` dicts.
    """

    def __init__(self, problems: list[dict[str, str]]) -> None:
        self._problems = problems
        self._idx = 0
        self._current_test = ""

    def reset(self) -> str:
        if self._idx >= len(self._problems):
            self._idx = 0
        problem = self._problems[self._idx]
        self._current_test = problem["test"]
        self._idx += 1
        return problem["prompt"]

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        # Try to extract a function and run the tests
        code = self._extract_function(action)
        success = self._run_tests(code, self._current_test)
        reward = 1.0 if success else 0.0
        return "", reward, True, {"code": code, "success": success}

    @staticmethod
    def _extract_function(response: str) -> str:
        """Extract Python function from model response."""
        # Look for code between ```python ... ``` or just use raw text
        if "```python" in response:
            start = response.index("```python") + 9
            end = response.index("```", start) if "```" in response[start:] else len(response)
            return response[start:end].strip()
        if "def " in response:
            start = response.index("def ")
            return response[start:].strip()
        return response.strip()

    @staticmethod
    def _run_tests(code: str, tests: str) -> bool:
        """Execute code + tests in a subprocess, return True if all pass."""
        # Rename the function to 'solution' for test compatibility
        full_code = code + "\nsolution = " + (
            code.split("(")[0].replace("def ", "") if "def " in code else "lambda: None"
        ) + "\n" + tests

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    timeout=5,
                )
                return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Code RL training")
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

    env = CodeEnv(SAMPLE_PROBLEMS)
    sample_params = SamplingParams(max_tokens=256, temperature=0.8)

    for step in range(args.steps):
        trajectories = [
            rollout(env, sc, run.tokenizer, sample_params)
            for _ in range(args.rollouts_per_step)
        ]

        avg_reward = sum(t.total_reward for t in trajectories) / len(trajectories)
        data = compute_advantages(trajectories, run.tokenizer, method="grpo")
        if data:
            result = tc.forward_backward(data, loss_fn=CrossEntropyLoss())
            tc.optim_step(AdamParams(lr=args.lr))
            print(
                f"Step {step + 1}/{args.steps} | "
                f"Loss: {result.loss:.4f} | "
                f"Avg Reward: {avg_reward:.2f}"
            )

    tc.save_weights("./code_rl_output")
    print("Done!")


if __name__ == "__main__":
    main()
