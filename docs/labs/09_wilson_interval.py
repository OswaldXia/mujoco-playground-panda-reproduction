"""Compute two-sided Wilson confidence intervals without SciPy."""

from __future__ import annotations

import math


def wilson_interval(
    successes: int, episodes: int, z: float = 1.96
) -> tuple[float, float]:
  if episodes <= 0 or not 0 <= successes <= episodes:
    raise ValueError("require 0 <= successes <= episodes and episodes > 0")
  p = successes / episodes
  denominator = 1.0 + z * z / episodes
  center = (p + z * z / (2.0 * episodes)) / denominator
  half_width = z * math.sqrt(
      p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes)
  ) / denominator
  return center - half_width, center + half_width


def main() -> None:
  cases = ((62, 64), (964, 1024), (990, 1024))
  print(" successes/episodes  estimate  Wilson 95% interval  width")
  for successes, episodes in cases:
    lower, upper = wilson_interval(successes, episodes)
    rate = successes / episodes
    print(
        f" {successes:4d}/{episodes:<4d}       {rate:7.3%}  "
        f"[{lower:7.3%}, {upper:7.3%}]  {upper-lower:7.3%}"
    )


if __name__ == "__main__":
  main()
