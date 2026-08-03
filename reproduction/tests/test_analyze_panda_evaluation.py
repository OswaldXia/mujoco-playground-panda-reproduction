"""Tests for position-stratified Panda evaluation analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "analyze_panda_evaluation.py"
SPEC = importlib.util.spec_from_file_location("analyze_panda_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AnalyzePandaEvaluationTest(unittest.TestCase):

  def test_bins_include_right_edge_and_compute_rates(self):
    records = [
        {"initial_box_position": [0.63, -0.05, 0.0], "success": False},
        {"initial_box_position": [0.63, -0.01, 0.0], "success": True},
        {"initial_box_position": [0.63, 0.01, 0.0], "success": True},
        {"initial_box_position": [0.63, 0.05, 0.0], "success": False},
    ]
    bins = MODULE.bin_episode_records(records, bins=2, y_min=-0.05, y_max=0.05)
    self.assertEqual([item["episodes"] for item in bins], [2, 2])
    self.assertEqual([item["successes"] for item in bins], [1, 1])
    self.assertEqual([item["success_rate"] for item in bins], [0.5, 0.5])

  def test_empty_bin_has_no_rate_or_interval(self):
    records = [
        {"initial_box_position": [0.63, -0.04, 0.0], "success": True},
    ]
    bins = MODULE.bin_episode_records(records, bins=2, y_min=-0.05, y_max=0.05)
    self.assertIsNone(bins[1]["success_rate"])
    self.assertIsNone(bins[1]["success_rate_wilson_95"])

  def test_wilson_interval_contains_zero_success_boundary(self):
    low, high = MODULE.wilson_interval(0, 3)
    self.assertEqual(low, 0.0)
    self.assertGreater(high, 0.0)

  def test_invalid_position_range_is_rejected(self):
    with self.assertRaises(ValueError):
      MODULE.bin_episode_records([], bins=2, y_min=0.05, y_max=-0.05)

  def test_writers_create_complete_analysis_artifacts(self):
    records = [
        {
            "episode_id": "101:0000",
            "seed": 101,
            "environment_index": 0,
            "initial_box_position": [0.63, -0.04, 0.0],
            "success": False,
            "success_metric": 0.0,
            "reward": 1.0,
            "episode_length": 200,
            "failure_class": "reached_no_lift",
            "trajectory": {
                "gripper_aperture_at_first_reached": 0.04,
                "ever_bilateral_finger_box_contact": False,
                "reach_to_lift_steps": -1,
            },
        },
        {
            "episode_id": "101:0001",
            "seed": 101,
            "environment_index": 1,
            "initial_box_position": [0.63, 0.04, 0.0],
            "success": True,
            "success_metric": 1.0,
            "reward": 10.0,
            "episode_length": 70,
        },
    ]
    bins = MODULE.bin_episode_records(records, 2, -0.05, 0.05)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      MODULE.write_episode_csv(root / "episodes.csv", records)
      MODULE.write_bin_csv(root / "bins.csv", bins)
      MODULE.write_position_plot(root / "plot.png", bins)
      self.assertGreater((root / "episodes.csv").stat().st_size, 0)
      self.assertGreater((root / "bins.csv").stat().st_size, 0)
      self.assertGreater((root / "plot.png").stat().st_size, 0)
      episode_rows = (root / "episodes.csv").read_text(encoding="utf-8")
      self.assertIn("gripper_aperture_at_first_reached", episode_rows)
      self.assertIn("ever_bilateral_finger_box_contact", episode_rows)
      self.assertIn("reached_no_lift", episode_rows)

  def test_loader_rejects_aggregate_mismatch(self):
    report = {
        "evaluation": {
            "aggregate": {"episodes": 2, "successes": 1},
            "episode_records": [{
                "episode_id": "101:0000",
                "seed": 101,
                "environment_index": 0,
                "initial_box_position": [0.63, 0.0, 0.0],
                "success": True,
                "reward": 1.0,
                "episode_length": 10,
            }],
        }
    }
    with tempfile.TemporaryDirectory() as directory:
      report_path = Path(directory) / "report.json"
      report_path.write_text(json.dumps(report), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "aggregate.episodes"):
        MODULE.load_episode_records(report_path)

  def test_trajectory_classification_is_exported(self):
    report = {
        "generated_at_utc": "2026-08-03T00:00:00+00:00",
        "evaluation": {
            "trajectory_diagnostics": {
                "thresholds_meters": {"approach_distance": 0.03},
                "notes": {"ever_hand_box_collision": "not grasp contact"},
                "failure_classification": {
                    "total_failures": 1,
                    "counts": {"reached_no_lift": 1},
                },
            }
        },
    }
    failures = [{
        "episode_id": "101:0000",
        "success": False,
        "failure_class": "reached_no_lift",
        "trajectory": {"ever_reached_box": True, "ever_lifted": False},
    }]
    artifact = MODULE.build_failure_classification_artifact(report, failures)
    self.assertIsNotNone(artifact)
    self.assertEqual(artifact["summary"]["total_failures"], 1)
    self.assertEqual(
        artifact["failure_cases"][0]["failure_class"], "reached_no_lift"
    )

  def test_grasp_acquisition_comparison_separates_outcomes(self):
    def record(success, bilateral, aperture, latency):
      return {
          "success": success,
          "reward": 10.0 if success else 5.0,
          "trajectory": {
              "gripper_aperture_at_first_reached": aperture,
              "close_command_fraction_until_reach": 0.5,
              "close_command_at_first_reached": True,
              "ever_bilateral_finger_box_contact": bilateral,
              "bilateral_contact_at_first_reached": bilateral,
              "ever_lifted": success,
              "reach_to_lift_steps": latency,
          },
      }

    comparison = MODULE.build_grasp_acquisition_comparison([
        record(True, True, 0.04, 30),
        record(True, True, 0.06, 40),
        record(False, False, 0.02, -1),
    ])
    self.assertIsNotNone(comparison)
    self.assertEqual(comparison["success"]["ever_bilateral_contact_rate"], 1.0)
    self.assertEqual(comparison["failure"]["ever_bilateral_contact_rate"], 0.0)
    self.assertEqual(
        comparison["success"]["mean_gripper_aperture_at_reach"], 0.05
    )
    self.assertEqual(comparison["success"]["mean_reach_to_lift_steps"], 35.0)


if __name__ == "__main__":
  unittest.main()
