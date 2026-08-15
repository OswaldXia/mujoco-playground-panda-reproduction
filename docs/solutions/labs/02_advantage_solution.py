"""Reference solution for return, GAE, and PPO clipping."""

from __future__ import annotations

import numpy as np


def discounted_returns(
    rewards: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
  result = np.zeros_like(rewards, dtype=float)
  running = 0.0
  for index in range(len(rewards) - 1, -1, -1):
    running = rewards[index] + gamma * (1.0 - dones[index]) * running
    result[index] = running
  return result


def generalized_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
  result = np.zeros_like(rewards, dtype=float)
  running = 0.0
  for index in range(len(rewards) - 1, -1, -1):
    active = 1.0 - dones[index]
    delta = rewards[index] + gamma * active * values[index + 1] - values[index]
    running = delta + gamma * gae_lambda * active * running
    result[index] = running
  return result


def clipped_objective(
    ratio: np.ndarray, advantage: np.ndarray, epsilon: float
) -> np.ndarray:
  raw = ratio * advantage
  clipped = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
  return np.minimum(raw, clipped)


def main() -> None:
  rewards = np.array([0.0, 1.0])
  dones = np.array([0.0, 1.0])
  values = np.array([0.4, 0.6, 0.0])
  np.testing.assert_allclose(discounted_returns(rewards, dones, 0.9), [0.9, 1.0])
  np.testing.assert_allclose(
      generalized_advantage(rewards, values, dones, 0.9, 0.95),
      [0.482, 0.4],
      atol=1e-9,
  )
  np.testing.assert_allclose(
      clipped_objective(
          np.array([1.5, 0.5]), np.array([1.0, -1.0]), 0.2
      ),
      [1.2, -0.8],
  )
  print("PASS: advantage solution")


if __name__ == "__main__":
  main()
