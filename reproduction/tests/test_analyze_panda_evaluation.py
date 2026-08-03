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


if __name__ == "__main__":
  unittest.main()
