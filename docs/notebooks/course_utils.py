"""Shared, side-effect-free helpers for the Panda course notebooks."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def find_repo_root(start: Path | None = None) -> Path:
  """Finds the repository root from a notebook or repository working directory."""
  start = (start or Path.cwd()).resolve()
  for candidate in (start, *start.parents):
    if (candidate / "pyproject.toml").is_file() and (
        candidate / "docs" / "course" / "README.md"
    ).is_file():
      return candidate
  raise RuntimeError(
      "Repository root not found. Start Jupyter with "
      "./reproduction/start_course_notebooks.sh"
  )


def assert_course_kernel(root: Path) -> None:
  """Rejects a system kernel when the repository virtual environment exists."""
  expected = (root / ".venv").resolve()
  active_prefix = Path(sys.prefix).resolve()
  if expected.exists() and active_prefix != expected:
    raise RuntimeError(
        f"Wrong kernel: {sys.executable}. Expected the project environment under "
        f"{expected}. Restart with ./reproduction/start_course_notebooks.sh"
    )


def rotation_z(theta_degrees: float) -> np.ndarray:
  theta = math.radians(theta_degrees)
  c, s = math.cos(theta), math.sin(theta)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
  transform = np.eye(4)
  transform[:3, :3] = rotation
  transform[:3, 3] = translation
  return transform


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
  homogeneous = np.concatenate([np.asarray(point, dtype=float), np.ones(1)])
  return (transform @ homogeneous)[:3]


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
  """Computes GAE; values contains one extra bootstrap element."""
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


def wilson_interval(
    successes: int, episodes: int, z: float = 1.96
) -> tuple[float, float]:
  if episodes <= 0 or not 0 <= successes <= episodes:
    raise ValueError("successes and episodes must define a valid binomial sample")
  p = successes / episodes
  denominator = 1.0 + z * z / episodes
  center = (p + z * z / (2.0 * episodes)) / denominator
  radius = z * math.sqrt(
      p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes)
  ) / denominator
  return center - radius, center + radius


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def trajectory_events(record: dict[str, Any]) -> list[tuple[str, int]]:
  """Returns present trajectory milestones in causal order."""
  fields = (
      ("reached", "first_reached_step"),
      ("lifted", "first_lift_step"),
      ("success", "first_success_step"),
  )
  events = []
  for label, field in fields:
    step = int(record[field])
    if step >= 0:
      events.append((label, step))
  return events
