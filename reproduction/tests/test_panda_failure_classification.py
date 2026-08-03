"""Tests for trajectory-level Panda failure classification."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "panda_failure_classification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "panda_failure_classification", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PandaFailureClassificationTest(unittest.TestCase):

  def test_stage_order_yields_mutually_exclusive_classes(self):
    cases = (
        ({"ever_out_of_bounds_or_invalid": True}, "out_of_bounds_or_invalid"),
        ({"ever_approached": False}, "never_approached"),
        ({"ever_approached": True}, "approached_not_reached"),
        (
            {"ever_approached": True, "ever_reached_box": True},
            "reached_no_lift",
        ),
        (
            {
                "ever_approached": True,
                "ever_lifted": True,
                "lifted_then_dropped": True,
            },
            "lifted_then_dropped",
        ),
        (
            {
                "ever_approached": True,
                "ever_lifted": True,
            },
            "lifted_timeout",
        ),
    )
    for trajectory, expected in cases:
      with self.subTest(expected=expected):
        self.assertEqual(MODULE.classify_failure(trajectory), expected)

  def test_summary_counts_only_failures_and_keeps_empty_classes(self):
    records = [
        {"episode_id": "1:0000", "success": True, "failure_class": None},
        {
            "episode_id": "1:0001",
            "success": False,
            "failure_class": "never_approached",
        },
        {
            "episode_id": "1:0002",
            "success": False,
            "failure_class": "never_approached",
        },
        {
            "episode_id": "1:0003",
            "success": False,
            "failure_class": "reached_no_lift",
        },
    ]
    summary = MODULE.summarize_failure_classes(records)
    self.assertEqual(summary["total_failures"], 3)
    self.assertEqual(summary["counts"]["never_approached"], 2)
    self.assertEqual(summary["counts"]["lifted_timeout"], 0)
    self.assertEqual(summary["dominant_class"], "never_approached")
    self.assertEqual(
        summary["representative_episode_ids"]["reached_no_lift"],
        ["1:0003"],
    )


if __name__ == "__main__":
  unittest.main()
