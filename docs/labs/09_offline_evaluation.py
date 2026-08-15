"""Recompute formal evaluation decisions from committed compact evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "reproduction" / "results" / "linux-guide-free-left-trajectory-analysis.json"
)


def wilson(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
  p = successes / episodes
  denominator = 1.0 + z * z / episodes
  center = (p + z * z / (2.0 * episodes)) / denominator
  radius = z * math.sqrt(
      p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes)
  ) / denominator
  return center - radius, center + radius


def analyze(report: dict) -> dict[str, object]:
  aggregate = report["aggregate"]
  per_seed = report["per_seed"]
  episodes = int(report["protocol"]["episodes"])
  successes = int(aggregate["successes"])
  failures = int(aggregate["failures"])
  if successes + failures != episodes:
    raise ValueError("aggregate successes + failures does not equal episodes")
  if sum(int(row["successes"]) for row in per_seed) != successes:
    raise ValueError("per-seed successes do not add up to aggregate")
  if sum(int(row["episodes"]) for row in per_seed) != episodes:
    raise ValueError("per-seed episodes do not add up to aggregate")
  interval = wilson(successes, episodes)
  recorded = aggregate["success_rate_wilson_95"]
  if max(abs(interval[i] - float(recorded[i])) for i in (0, 1)) > 1e-6:
    raise ValueError("recomputed Wilson interval differs from recorded evidence")
  worst_seed = min(int(row["successes"]) / int(row["episodes"]) for row in per_seed)
  return {
      "episodes": episodes,
      "successes": successes,
      "success_rate": successes / episodes,
      "wilson_95": interval,
      "worst_seed_success_rate": worst_seed,
      "aggregate_target": 0.95,
      "aggregate_passed": successes / episodes >= 0.95,
      "worst_seed_target": 0.90,
      "worst_seed_passed": worst_seed >= 0.90,
      "overall_passed": successes / episodes >= 0.95 and worst_seed >= 0.90,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()
  result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
  if args.json:
    print(json.dumps(result, indent=2, sort_keys=True))
    return
  try:
    source = args.input.resolve().relative_to(ROOT)
  except ValueError:
    source = args.input.resolve()
  low, high = result["wilson_95"]
  print("\nOffline evaluation audit")
  print("=" * 68)
  print(f"  Source          {source}")
  print(f"  Success         {result['successes']}/{result['episodes']}")
  print(f"  Rate            {result['success_rate']:.2%}")
  print(f"  Wilson 95%      [{low:.2%}, {high:.2%}]")
  print(f"  Worst seed      {result['worst_seed_success_rate']:.2%}")
  print("-" * 68)
  print(f"  Aggregate gate  {'PASS' if result['aggregate_passed'] else 'FAIL'} (>=95%)")
  print(f"  Worst-seed gate {'PASS' if result['worst_seed_passed'] else 'FAIL'} (>=90%)")
  print(f"  Decision        {'PASS' if result['overall_passed'] else 'FAIL'}")
  print("\nPASS: counts, seeds, interval, and fixed decision rule are consistent\n")


if __name__ == "__main__":
  main()
