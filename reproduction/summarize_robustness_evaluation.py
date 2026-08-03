"""Compare original, left-side, and hard-bin Panda robustness evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


def hypergeometric_probability(a: int, b: int, c: int, d: int) -> float:
  total = a + b + c + d
  return (
      math.comb(a + b, a)
      * math.comb(c + d, c)
      / math.comb(total, a + c)
  )


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
  """Returns the probability-ordering two-sided Fisher exact p-value."""
  row_one = a + b
  row_two = c + d
  column_one = a + c
  lower = max(0, column_one - row_two)
  upper = min(row_one, column_one)
  observed = hypergeometric_probability(a, b, c, d)
  probability = 0.0
  for candidate_a in range(lower, upper + 1):
    candidate_b = row_one - candidate_a
    candidate_c = column_one - candidate_a
    candidate_d = row_two - candidate_c
    candidate = hypergeometric_probability(
        candidate_a, candidate_b, candidate_c, candidate_d
    )
    if candidate <= observed + 1e-15:
      probability += candidate
  return min(1.0, probability)


def load_result(path: Path) -> dict[str, Any]:
  report = json.loads(path.read_text(encoding="utf-8"))
  evaluation = report["evaluation"]
  aggregate = evaluation["aggregate"]
  return {
      "source_report": str(path.resolve()),
      "checkpoint_step": report.get("checkpoint_step"),
      "sampling": evaluation.get("initial_y_sampling"),
      "episodes": aggregate["episodes"],
      "successes": aggregate["successes"],
      "failures": aggregate["failures"],
      "success_rate": aggregate["success_rate"],
      "success_rate_wilson_95": aggregate["success_rate_wilson_95"],
      "worst_seed_success_rate": aggregate["worst_seed_success_rate"],
      "mean_reward": aggregate["mean_reward"],
  }


def build_summary(
    original: dict[str, Any],
    left: dict[str, Any],
    hard: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
  baseline_groups = baseline["grouped_results"]
  baseline_left = baseline_groups["y_below_negative_0_02"]
  baseline_hard = baseline_groups[
      "hard_bin_negative_0_03_to_negative_0_02"
  ]
  criteria = {
      "original_success_rate_at_least_0_95": original["success_rate"] >= 0.95,
      "original_worst_seed_at_least_0_90": (
          original["worst_seed_success_rate"] >= 0.90
      ),
      "left_success_rate_at_least_0_95": left["success_rate"] >= 0.95,
      "left_worst_seed_at_least_0_90": left["worst_seed_success_rate"] >= 0.90,
      "hard_success_rate_at_least_0_93": hard["success_rate"] >= 0.93,
      "hard_worst_seed_at_least_0_85": hard["worst_seed_success_rate"] >= 0.85,
  }
  left_p_value = fisher_exact_two_sided(
      left["successes"],
      left["failures"],
      baseline_left["successes"],
      baseline_left["failures"],
  )
  hard_p_value = fisher_exact_two_sided(
      hard["successes"],
      hard["failures"],
      baseline_hard["successes"],
      baseline_hard["failures"],
  )
  return {
      "schema_version": 1,
      "experiment": "Panda targeted left-y robustness fine-tune",
      "evaluations": {"original": original, "left": left, "hard": hard},
      "baseline": {
          "left": baseline_left,
          "hard": baseline_hard,
      },
      "improvement": {
          "left_success_rate_absolute": (
              left["success_rate"] - baseline_left["success_rate"]
          ),
          "left_fisher_two_sided_p_value": left_p_value,
          "hard_success_rate_absolute": (
              hard["success_rate"] - baseline_hard["success_rate"]
          ),
          "hard_fisher_two_sided_p_value": hard_p_value,
      },
      "acceptance": {
          "criteria": criteria,
          "passed": all(criteria.values()),
      },
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--original", type=Path, required=True)
  parser.add_argument("--left", type=Path, required=True)
  parser.add_argument("--hard", type=Path, required=True)
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--require-pass", action="store_true")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  summary = build_summary(
      load_result(args.original),
      load_result(args.left),
      load_result(args.hard),
      json.loads(args.baseline.read_text(encoding="utf-8")),
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

  print("\nRobustness regression decision")
  for name, result in summary["evaluations"].items():
    print(
        f"  {name:<10}{result['successes']}/{result['episodes']} "
        f"({result['success_rate']:.2%}), "
        f"worst seed {result['worst_seed_success_rate']:.2%}"
    )
  improvement = summary["improvement"]
  print(
      f"  {'left gain':<10}{improvement['left_success_rate_absolute']:+.2%} "
      f"(Fisher p={improvement['left_fisher_two_sided_p_value']:.4g})"
  )
  print(
      f"  {'hard gain':<10}{improvement['hard_success_rate_absolute']:+.2%} "
      f"(Fisher p={improvement['hard_fisher_two_sided_p_value']:.4g})"
  )
  decision = "PASS" if summary["acceptance"]["passed"] else "FAIL"
  print(f"  {'decision':<10}{decision}")
  print(f"  {'summary':<10}{args.output.resolve()}")
  if args.require_pass and not summary["acceptance"]["passed"]:
    sys.exit(1)


if __name__ == "__main__":
  main()
