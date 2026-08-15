"""Print the PPO clipped surrogate for a small set of examples."""

from __future__ import annotations

import argparse

import numpy as np


def clipped_surrogate(
    ratio: np.ndarray, advantage: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns unclipped, clipped, and conservative PPO sample objectives."""
  unclipped = ratio * advantage
  clipped = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
  return unclipped, clipped, np.minimum(unclipped, clipped)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--epsilon", type=float, default=0.2)
  args = parser.parse_args()
  if not 0.0 < args.epsilon < 1.0:
    raise SystemExit("--epsilon must be between zero and one")

  ratios = np.array([0.5, 1.0, 1.5, 0.5, 1.0, 1.5])
  advantages = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
  raw, clipped, objective = clipped_surrogate(
      ratios, advantages, args.epsilon
  )

  print(f"epsilon={args.epsilon:.2f}")
  print(" ratio  advantage  raw     clipped  objective")
  for values in zip(ratios, advantages, raw, clipped, objective):
    print(" %5.2f  %+9.2f  %+6.2f  %+7.2f  %+9.2f" % values)


if __name__ == "__main__":
  main()
