"""Reference solution for the 3-D homogeneous-transform exercise."""

from __future__ import annotations

import math

import numpy as np


def rotation_z(theta_degrees: float) -> np.ndarray:
  theta = math.radians(theta_degrees)
  c, s = math.cos(theta), math.sin(theta)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
  result = np.eye(4)
  result[:3, :3] = rotation
  result[:3, 3] = translation
  return result


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
  homogeneous = np.concatenate([point, np.ones(1)])
  return (transform @ homogeneous)[:3]


def main() -> None:
  world_from_base = rigid_transform(
      rotation_z(90.0), np.array([2.0, 1.0, 0.5])
  )
  base_from_tip = rigid_transform(
      np.eye(3), np.array([1.0, 0.0, 0.25])
  )
  world_from_tip = world_from_base @ base_from_tip
  np.testing.assert_allclose(
      transform_point(world_from_tip, np.zeros(3)),
      [2.0, 2.0, 0.75],
      atol=1e-9,
  )
  np.testing.assert_allclose(
      world_from_tip @ np.linalg.inv(world_from_tip), np.eye(4), atol=1e-9
  )
  print("PASS: 3-D transform solution")


if __name__ == "__main__":
  main()
