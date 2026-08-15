"""Complete three 3-D transform functions, then run this file.

Hints:
  Level 1: a point uses homogeneous last coordinate 1; a direction uses 0.
  Level 2: rotation_z contains cos/sin in its top-left 2x2 block.
  Level 3: compose world<-base and base<-tip in that left-to-right order.
"""

from __future__ import annotations

import math

import numpy as np


def rotation_z(theta_degrees: float) -> np.ndarray:
  """Returns a 3x3 rotation about z."""
  del theta_degrees
  raise NotImplementedError("TODO: implement rotation_z")


def rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
  """Returns a 4x4 homogeneous transform."""
  del rotation, translation
  raise NotImplementedError("TODO: implement rigid_transform")


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
  """Transforms a 3-D point and returns its first three coordinates."""
  del transform, point
  raise NotImplementedError("TODO: implement transform_point")


def main() -> None:
  world_from_base = rigid_transform(
      rotation_z(90.0), np.array([2.0, 1.0, 0.5])
  )
  base_from_tip = rigid_transform(
      np.eye(3), np.array([1.0, 0.0, 0.25])
  )
  world_from_tip = world_from_base @ base_from_tip
  tip_origin_world = transform_point(world_from_tip, np.zeros(3))

  np.testing.assert_allclose(tip_origin_world, [2.0, 2.0, 0.75], atol=1e-9)
  np.testing.assert_allclose(
      world_from_tip @ np.linalg.inv(world_from_tip), np.eye(4), atol=1e-9
  )
  print("PASS: 3-D transform composition and inverse are consistent")


if __name__ == "__main__":
  main()
