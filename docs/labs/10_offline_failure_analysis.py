"""Validate trajectory-level failure evidence without a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    ROOT / "reproduction" / "results" / "linux-guide-free-left-trajectory-analysis.json"
)
DEFAULT_FIXTURE = ROOT / "docs" / "data" / "guide-free-left-episodes-fixture.json"


def analyze(summary: dict, fixture: dict) -> dict[str, object]:
  failures = int(summary["aggregate"]["failures"])
  counts = summary["failure_classification"]["counts"]
  if sum(int(value) for value in counts.values()) != failures:
    raise ValueError("failure classes are not exhaustive")
  dominant = max(counts, key=counts.get)
  if dominant != summary["failure_classification"]["dominant_class"]:
    raise ValueError("recorded dominant failure class is inconsistent")
  if fixture["fixture_role"] != "teaching_only_curated_examples_not_a_rate_sample":
    raise ValueError("episode fixture is missing its non-representative warning")
  examples = fixture["curated_episode_examples"]
  if any(row["success"] and row["failure_class"] is not None for row in examples):
    raise ValueError("successful example has a failure class")
  failure_examples = [row for row in examples if not row["success"]]
  if any(not row["ever_bilateral_contact"] for row in failure_examples):
    raise ValueError("fixture no longer supports the documented contact finding")
  comparison = summary["grasp_acquisition_comparison"]
  return {
      "failures": failures,
      "dominant_class": dominant,
      "dominant_count": int(counts[dominant]),
      "dominant_share": int(counts[dominant]) / failures,
      "success_bilateral_contact_rate": comparison["success"][
          "ever_bilateral_contact_rate"
      ],
      "failure_bilateral_contact_rate": comparison["failure"][
          "ever_bilateral_contact_rate"
      ],
      "curated_examples": len(examples),
      "mechanism": "post-contact reach-to-lift stability",
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
  parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
  args = parser.parse_args()
  result = analyze(
      json.loads(args.summary.read_text(encoding="utf-8")),
      json.loads(args.fixture.read_text(encoding="utf-8")),
  )
  print("\nOffline trajectory failure audit")
  print("=" * 68)
  print(f"  Exhaustive failures  {result['failures']}")
  print(
      f"  Dominant class       {result['dominant_class']} "
      f"({result['dominant_count']}/{result['failures']}, {result['dominant_share']:.2%})"
  )
  print(f"  Contact / successes  {result['success_bilateral_contact_rate']:.2%}")
  print(f"  Contact / failures   {result['failure_bilateral_contact_rate']:.2%}")
  print(f"  Curated examples     {result['curated_examples']} (not a rate sample)")
  print("-" * 68)
  print(f"  Supported next focus {result['mechanism']}")
  print("\nPASS: classes are exhaustive and the mechanism follows the evidence\n")


if __name__ == "__main__":
  main()
