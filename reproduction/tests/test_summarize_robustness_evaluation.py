"""Tests for the controlled Panda robustness decision."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_robustness_evaluation.py"
SPEC = importlib.util.spec_from_file_location("summarize_robustness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SummarizeRobustnessEvaluationTest(unittest.TestCase):

  def test_fisher_exact_matches_known_reference(self):
    result = MODULE.fisher_exact_two_sided(265, 25, 723, 11)
    self.assertAlmostEqual(result, 2.1726974965935027e-07, places=15)

  def test_acceptance_requires_all_three_distributions(self):
    def result(rate, worst, episodes=1000):
      successes = round(rate * episodes)
      return {
          "success_rate": rate,
          "worst_seed_success_rate": worst,
          "episodes": episodes,
          "successes": successes,
          "failures": episodes - successes,
      }

    baseline = {
        "grouped_results": {
            "y_below_negative_0_02": {
                "successes": 265,
                "failures": 25,
                "success_rate": 265 / 290,
            },
            "hard_bin_negative_0_03_to_negative_0_02": {
                "successes": 67,
                "failures": 9,
                "success_rate": 67 / 76,
            },
        }
    }
    passing = MODULE.build_summary(
        result(0.96, 0.94),
        result(0.96, 0.93),
        result(0.94, 0.90),
        baseline,
    )
    self.assertTrue(passing["acceptance"]["passed"])
    failing = MODULE.build_summary(
        result(0.94, 0.93),
        result(0.96, 0.93),
        result(0.94, 0.90),
        baseline,
    )
    self.assertFalse(failing["acceptance"]["passed"])


if __name__ == "__main__":
  unittest.main()
