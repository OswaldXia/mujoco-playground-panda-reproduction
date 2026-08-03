"""Analyze Panda held-out evaluation outcomes by initial cube position."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError("value must be a positive integer")
  return parsed


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
  if total <= 0:
    return 0.0, 0.0
  z = 1.959963984540054
  rate = successes / total
  denominator = 1.0 + z * z / total
  center = (rate + z * z / (2.0 * total)) / denominator
  margin = (
      z
      * math.sqrt(
          rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
      )
      / denominator
  )
  return (
      max(0.0, min(rate, center - margin)),
      min(1.0, max(rate, center + margin)),
  )


def load_episode_records(report_path: Path) -> tuple[dict[str, Any], list[dict]]:
  report = json.loads(report_path.read_text(encoding="utf-8"))
  records = report.get("evaluation", {}).get("episode_records")
  if not isinstance(records, list) or not records:
    raise ValueError(
        "Report does not contain evaluation.episode_records. Generate it "
        "with schema version 2 of evaluate_panda_checkpoint.py."
    )
  required = {
      "episode_id",
      "seed",
      "environment_index",
      "initial_box_position",
      "success",
      "reward",
      "episode_length",
  }
  episode_ids = set()
  for index, record in enumerate(records):
    missing = required.difference(record)
    if missing:
      raise ValueError(f"Episode record {index} is missing: {sorted(missing)}")
    position = record["initial_box_position"]
    if not isinstance(position, list) or len(position) != 3:
      raise ValueError(f"Episode record {index} has an invalid box position")
    episode_id = record["episode_id"]
    if episode_id in episode_ids:
      raise ValueError(f"Duplicate episode_id: {episode_id}")
    episode_ids.add(episode_id)
  aggregate = report["evaluation"].get("aggregate", {})
  recorded_successes = sum(bool(record["success"]) for record in records)
  if aggregate.get("episodes") != len(records):
    raise ValueError("Episode record count does not match aggregate.episodes")
  if aggregate.get("successes") != recorded_successes:
    raise ValueError("Episode outcomes do not match aggregate.successes")
  return report, records


def bin_episode_records(
    records: list[dict], bins: int, y_min: float, y_max: float
) -> list[dict[str, Any]]:
  if bins <= 0:
    raise ValueError("bins must be positive")
  if y_min >= y_max:
    raise ValueError("y_min must be smaller than y_max")
  width = (y_max - y_min) / bins
  results = []
  for index in range(bins):
    lower = y_min + index * width
    upper = lower + width
    selected = []
    for record in records:
      y_value = float(record["initial_box_position"][1])
      in_bin = lower <= y_value < upper
      if index == bins - 1:
        in_bin = lower <= y_value <= upper
      if in_bin:
        selected.append(record)
    total = len(selected)
    successes = sum(bool(record["success"]) for record in selected)
    low, high = wilson_interval(successes, total)
    results.append({
        "bin_index": index,
        "y_lower": lower,
        "y_upper": upper,
        "episodes": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total if total else None,
        "success_rate_wilson_95": [low, high] if total else None,
    })
  if sum(item["episodes"] for item in results) != len(records):
    raise ValueError(
        f"At least one initial y position falls outside [{y_min}, {y_max}]"
    )
  return results


def write_episode_csv(path: Path, records: list[dict]) -> None:
  fields = (
      "episode_id",
      "seed",
      "environment_index",
      "initial_x",
      "initial_y",
      "initial_z",
      "success",
      "success_metric",
      "reward",
      "episode_length",
  )
  with path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for record in records:
      position = record["initial_box_position"]
      writer.writerow({
          "episode_id": record["episode_id"],
          "seed": record["seed"],
          "environment_index": record["environment_index"],
          "initial_x": position[0],
          "initial_y": position[1],
          "initial_z": position[2],
          "success": int(bool(record["success"])),
          "success_metric": record.get("success_metric", ""),
          "reward": record["reward"],
          "episode_length": record["episode_length"],
      })


def write_bin_csv(path: Path, bins: list[dict]) -> None:
  with path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=bins[0].keys())
    writer.writeheader()
    writer.writerows(bins)


def write_position_plot(
    path: Path, bins: list[dict], target_success_rate: float = 0.90
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  centers = [(item["y_lower"] + item["y_upper"]) / 2.0 for item in bins]
  widths = [item["y_upper"] - item["y_lower"] for item in bins]
  rates = [
      item["success_rate"] if item["success_rate"] is not None else 0.0
      for item in bins
  ]
  failures = [item["failures"] for item in bins]
  successes = [item["successes"] for item in bins]

  figure, (rate_axis, count_axis) = plt.subplots(
      2, 1, figsize=(9, 6), sharex=True, height_ratios=(2.2, 1.0)
  )
  colors = [
      "#2a9d8f" if rate >= target_success_rate else "#e76f51"
      for rate in rates
  ]
  rate_axis.bar(centers, rates, width=widths, color=colors, edgecolor="white")
  lower_errors = []
  upper_errors = []
  for rate, item in zip(rates, bins, strict=True):
    interval = item["success_rate_wilson_95"]
    lower_errors.append(max(0.0, rate - interval[0]) if interval else 0.0)
    upper_errors.append(max(0.0, interval[1] - rate) if interval else 0.0)
  rate_axis.errorbar(
      centers,
      rates,
      yerr=[lower_errors, upper_errors],
      fmt="none",
      ecolor="#1d3557",
      capsize=3,
      linewidth=1,
  )
  rate_axis.axhline(
      target_success_rate,
      color="#264653",
      linestyle="--",
      label=f"{target_success_rate:.0%} target",
  )
  rate_axis.set_ylim(0.0, 1.05)
  rate_axis.set_ylabel("Success rate")
  rate_axis.set_title("Panda success by initial cube y position")
  rate_axis.grid(axis="y", alpha=0.25)
  rate_axis.legend(loc="lower right")

  count_axis.bar(
      centers,
      failures,
      width=widths,
      color="#e63946",
      label="Failures",
  )
  count_axis.bar(
      centers,
      successes,
      width=widths,
      bottom=failures,
      color="#457b9d",
      label="Successes",
  )
  count_axis.set_xlabel("Initial cube y position (m)")
  count_axis.set_ylabel("Count")
  count_axis.grid(axis="y", alpha=0.25)
  count_axis.legend(loc="upper right")
  figure.tight_layout()
  figure.savefig(path, dpi=180)
  plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Create position-stratified analysis from a Panda report."
  )
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--bins", type=positive_int, default=10)
  parser.add_argument("--y-min", type=float)
  parser.add_argument("--y-max", type=float)
  parser.add_argument("--overwrite", action="store_true")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  report_path = args.report.expanduser().resolve()
  output_dir = (
      args.output_dir.expanduser().resolve()
      if args.output_dir
      else report_path.with_name(f"{report_path.stem}-analysis")
  )
  if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
    raise FileExistsError(
        f"Analysis directory is not empty; pass --overwrite: {output_dir}"
    )
  output_dir.mkdir(parents=True, exist_ok=True)

  report, records = load_episode_records(report_path)
  if (args.y_min is None) != (args.y_max is None):
    raise ValueError("--y-min and --y-max must be supplied together")
  report_range = report["evaluation"].get("initial_y_sampling", {}).get(
      "range", [-0.05, 0.05]
  )
  y_min = args.y_min if args.y_min is not None else float(report_range[0])
  y_max = args.y_max if args.y_max is not None else float(report_range[1])
  bins = bin_episode_records(records, args.bins, y_min, y_max)
  failures = [record for record in records if not record["success"]]
  failures.sort(key=lambda item: (item["initial_box_position"][1], item["seed"]))

  episode_csv = output_dir / "episodes.csv"
  bin_csv = output_dir / "position-bins.csv"
  failure_json = output_dir / "failure-cases.json"
  plot_path = output_dir / "position-success-rate.png"
  summary_path = output_dir / "analysis-summary.json"
  write_episode_csv(episode_csv, records)
  write_bin_csv(bin_csv, bins)
  failure_json.write_text(
      json.dumps({"count": len(failures), "cases": failures}, indent=2) + "\n",
      encoding="utf-8",
  )
  target_success_rate = report["evaluation"].get("acceptance", {}).get(
      "target_success_rate", 0.90
  )
  write_position_plot(plot_path, bins, target_success_rate)

  aggregate = report["evaluation"].get("aggregate", {})
  populated_bins = [item for item in bins if item["episodes"]]
  lowest_success_bin = min(
      populated_bins, key=lambda item: item["success_rate"]
  )
  summary = {
      "schema_version": 1,
      "source_report": str(report_path),
      "episodes": len(records),
      "successes": sum(bool(record["success"]) for record in records),
      "failures": len(failures),
      "aggregate_success_rate": aggregate.get("success_rate"),
      "position_axis": "initial_box_position.y",
      "position_range": [y_min, y_max],
      "lowest_success_bin": lowest_success_bin,
      "bins": bins,
      "artifacts": {
          "episodes_csv": episode_csv.name,
          "position_bins_csv": bin_csv.name,
          "failure_cases_json": failure_json.name,
          "position_plot": plot_path.name,
      },
  }
  summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

  print("\nPosition-stratified analysis complete")
  print(f"  {'Episodes':<22}{len(records)}")
  print(f"  {'Failures':<22}{len(failures)}")
  print(f"  {'Position bins':<22}{len(bins)} across [{y_min}, {y_max}]")
  print(
      f"  {'Lowest-success bin':<22}"
      f"[{lowest_success_bin['y_lower']:.3f}, "
      f"{lowest_success_bin['y_upper']:.3f}] = "
      f"{lowest_success_bin['success_rate']:.2%}"
  )
  print(f"  {'Plot':<22}{plot_path}")
  print(f"  {'Episode table':<22}{episode_csv}")
  print(f"  {'Failure cases':<22}{failure_json}")
  print(f"  {'Analysis summary':<22}{summary_path}")


if __name__ == "__main__":
  main()
