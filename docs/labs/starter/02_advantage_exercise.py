"""Implement discounted returns, GAE, and the PPO clipped objective.

Hints:
  Level 1: returns and GAE are computed from the final step backwards.
  Level 2: multiply every bootstrap term by (1 - done[t]).
  Level 3: PPO uses minimum(raw, clipped), including negative advantages.
"""

from __future__ import annotations

import numpy as np


def discounted_returns(
    rewards: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
  del rewards, dones, gamma
  raise NotImplementedError("TODO: implement discounted_returns")


def generalized_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
  """Values has one extra bootstrap element."""
  del rewards, values, dones, gamma, gae_lambda
  raise NotImplementedError("TODO: implement generalized_advantage")


def clipped_objective(
    ratio: np.ndarray, advantage: np.ndarray, epsilon: float
) -> np.ndarray:
  del ratio, advantage, epsilon
  raise NotImplementedError("TODO: implement clipped_objective")


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
  print("PASS: returns, GAE, and PPO clipping match hand calculations")


if __name__ == "__main__":
  main()
