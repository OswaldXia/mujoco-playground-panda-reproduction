"""Structural and numerical checks for Panda course notebooks."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "docs" / "notebooks"


def _load_module(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


VALIDATOR = _load_module(
    "validate_course_notebooks",
    ROOT / "reproduction" / "validate_course_notebooks.py",
)
UTILS = _load_module("course_utils", NOTEBOOK_DIR / "course_utils.py")


class CourseNotebooksTest(unittest.TestCase):

  def test_all_versioned_notebooks_have_clean_valid_structure(self) -> None:
    self.assertEqual(len(VALIDATOR.NOTEBOOKS), 5)
    for name in VALIDATOR.NOTEBOOKS:
      path = NOTEBOOK_DIR / name
      self.assertTrue(path.is_file(), name)
      self.assertEqual(VALIDATOR.validate_structure(path), [], name)
      notebook = json.loads(path.read_text(encoding="utf-8"))
      self.assertEqual(notebook["metadata"]["course"]["version"], "v0.10.1")

  def test_launcher_is_executable(self) -> None:
    launcher = ROOT / "reproduction" / "start_course_notebooks.sh"
    self.assertTrue(os.access(launcher, os.X_OK))
    text = launcher.read_text(encoding="utf-8")
    self.assertIn('--ServerApp.root_dir="${REPO_ROOT}"', text)
    self.assertIn('/lab/tree/docs/notebooks', text)

  def test_transform_helpers_match_hand_calculation(self) -> None:
    transform = UTILS.rigid_transform(
        UTILS.rotation_z(90.0), np.array([2.0, 1.0, 0.0])
    )
    np.testing.assert_allclose(
        UTILS.transform_point(transform, np.array([1.0, 0.0, 0.0])),
        [2.0, 2.0, 0.0],
        atol=1e-9,
    )

  def test_rl_helpers_match_hand_calculation(self) -> None:
    rewards = np.array([0.0, 1.0])
    dones = np.array([0.0, 1.0])
    values = np.array([0.4, 0.6, 0.0])
    np.testing.assert_allclose(
        UTILS.discounted_returns(rewards, dones, 0.9), [0.9, 1.0]
    )
    np.testing.assert_allclose(
        UTILS.generalized_advantage(rewards, values, dones, 0.9, 0.95),
        [0.482, 0.4],
    )
    np.testing.assert_allclose(
        UTILS.clipped_objective(
            np.array([1.5, 0.5]), np.array([1.0, -1.0]), 0.2
        ),
        [1.2, -0.8],
    )

  def test_wilson_helper_matches_committed_evidence(self) -> None:
    low, high = UTILS.wilson_interval(964, 1024)
    np.testing.assert_allclose(
        [low, high],
        [0.9253038995713028, 0.9542091734246192],
        atol=1e-6,
    )


if __name__ == "__main__":
  unittest.main()
