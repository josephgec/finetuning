"""RL environment abstractions and rollout utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from local_tinker.sampling_client import SamplingClient
from local_tinker.types import Datum, ModelInput, SamplingParams


# ---------------------------------------------------------------------------
# Environment abstractions
# ---------------------------------------------------------------------------


class Env(ABC):
    """Abstract RL environment interface."""

    @abstractmethod
    def reset(self) -> str:
        """Reset the environment and return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Take an action and return (observation, reward, done, info)."""
        ...


class MessageEnv(Env):
    """Chat-style environment where actions are assistant messages.

    Args:
        initial_message: The initial user message / prompt.
        reward_fn: Callable ``(response: str) -> float`` that scores the response.
    """

    def __init__(
        self,
        initial_message: str,
        reward_fn: Any,
    ) -> None:
        self._initial_message = initial_message
        self._reward_fn = reward_fn
        self._done = False

    def reset(self) -> str:
        self._done = False
        return self._initial_message

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        reward = float(self._reward_fn(action))
        self._done = True
        return "", reward, True, {}


class ProblemEnv(Env):
    """Problem-answer environment (e.g., math, code).

    Args:
        problems: List of ``{"question": str, "answer": str}`` dicts.
        extract_answer_fn: Optional callable to extract the answer from
            the model's response. Defaults to stripping whitespace.
    """

    def __init__(
        self,
        problems: list[dict[str, str]],
        extract_answer_fn: Any | None = None,
    ) -> None:
        self._problems = problems
        self._idx = 0
        self._current_answer = ""
        self._extract = extract_answer_fn or (lambda x: x.strip())

    def reset(self) -> str:
        if self._idx >= len(self._problems):
            self._idx = 0
        problem = self._problems[self._idx]
        self._current_answer = problem["answer"]
        self._idx += 1
        return problem["question"]

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        predicted = self._extract(action)
        correct = predicted == self._current_answer
        reward = 1.0 if correct else 0.0
        return "", reward, True, {"correct": correct, "predicted": predicted}


# ---------------------------------------------------------------------------
# Trajectory & rollout
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    """A single step in a trajectory."""

    observation: str
    action: str
    reward: float
    log_prob: float


@dataclass
class Trajectory:
    """A complete episode trajectory."""

    steps: list[TrajectoryStep] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def actions(self) -> list[str]:
        return [s.action for s in self.steps]


def rollout(
    env: Env,
    sampling_client: SamplingClient,
    tokenizer: Any,
    params: SamplingParams | None = None,
) -> Trajectory:
    """Run one complete episode: reset -> sample -> step -> done.

    Args:
        env: The RL environment.
        sampling_client: Client to generate actions.
        tokenizer: Tokenizer for encoding observations.
        params: Sampling parameters.

    Returns:
        A complete :class:`Trajectory`.
    """
    if params is None:
        params = SamplingParams(temperature=0.8)

    trajectory = Trajectory()
    obs = env.reset()
    done = False

    while not done:
        prompt = ModelInput.from_text(obs, tokenizer)
        response = sampling_client.sample(prompt, params)

        obs_next, reward, done, _info = env.step(response.text)

        avg_log_prob = (
            sum(response.log_probs) / len(response.log_probs)
            if response.log_probs
            else 0.0
        )

        trajectory.steps.append(
            TrajectoryStep(
                observation=obs,
                action=response.text,
                reward=reward,
                log_prob=avg_log_prob,
            )
        )
        obs = obs_next

    return trajectory


def compute_advantages(
    trajectories: list[Trajectory],
    tokenizer: Any,
    method: str = "grpo",
) -> list[Datum]:
    """Compute advantages from trajectories and package as training data.

    Args:
        trajectories: List of completed trajectories.
        tokenizer: Tokenizer for encoding.
        method: Advantage computation method. Currently supports ``"grpo"``
            (group-relative normalization).

    Returns:
        List of :class:`Datum` objects ready for ``forward_backward``.
    """
    rewards = torch.tensor([t.total_reward for t in trajectories])

    if method == "grpo":
        mean = rewards.mean()
        std = rewards.std().clamp(min=1e-8)
        advantages = (rewards - mean) / std
    else:
        advantages = rewards

    data: list[Datum] = []
    for traj, adv in zip(trajectories, advantages):
        for step in traj.steps:
            text = step.observation + step.action
            enc = tokenizer(text, add_special_tokens=True)
            input_ids = enc.input_ids
            # Labels = input_ids (train on entire sequence for RL)
            data.append(Datum(input_ids=input_ids, labels=input_ids))

    return data
