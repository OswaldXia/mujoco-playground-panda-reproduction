"""Minimal homogeneous-transform lab; requires only NumPy."""

from __future__ import annotations

import math

import numpy as np


def transform(theta_degrees: float, translation: tuple[float, float]) -> np.ndarray:
  """Returns a 3x3 homogeneous transform for a planar frame."""
  theta = math.radians(theta_degrees)
  c, s = math.cos(theta), math.sin(theta)
  return np.array([
      [c, -s, translation[0]],
      [s, c, translation[1]],
      [0.0, 0.0, 1.0],
  ])


def main() -> None:
  point = np.array([1.0, 0.0, 1.0])
  rotation = transform(90.0, (0.0, 0.0))
  translation = transform(0.0, (2.0, 1.0))

  rotate_then_translate = translation @ rotation @ point
  translate_then_rotate = rotation @ translation @ point
  r = rotation[:2, :2]

  np.testing.assert_allclose(rotate_then_translate[:2], [2.0, 2.0], atol=1e-9)
  np.testing.assert_allclose(translate_then_rotate[:2], [-1.0, 3.0], atol=1e-9)
  np.testing.assert_allclose(r.T @ r, np.eye(2), atol=1e-9)
  np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-9)

  print("point                       ", point[:2])
  print("rotate, then translate      ", rotate_then_translate[:2])
  print("translate, then rotate      ", translate_then_rotate[:2])
  print("rotation orthogonal / det=1 ", True)


if __name__ == "__main__":
  main()
