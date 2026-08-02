"""Select the strongest saved checkpoint from an evaluation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUCCESS_TAG = "eval/episode_reward/success"
REWARD_TAG = "eval/episode_reward"


def select_best(summary: dict) -> dict[str, float | int]:
  success_series = summary["series"][SUCCESS_TAG]
  reward_by_step = {
      int(point["step"]): float(point["value"])
      for point in summary["series"][REWARD_TAG]
  }
  if not success_series:
    raise ValueError(f"No values found for {SUCCESS_TAG}")
  best = max(
      success_series,
      key=lambda point: (
          float(point["value"]),
          reward_by_step.get(int(point["step"]), float("-inf")),
          -int(point["step"]),
      ),
  )
  step = int(best["step"])
  return {
      "step": step,
      "success": float(best["value"]),
      "reward": reward_by_step.get(step, float("nan")),
  }


def find_checkpoint(summary: dict, runs_dir: Path, step: int) -> Path:
  candidates = [
      path
      for path in runs_dir.rglob("checkpoints/*")
      if (
          path.is_dir()
          and path.name.isdigit()
          and int(path.name) == step
          and (path / "ppo_network_config.json").is_file()
      )
  ]
  if not candidates:
    raise FileNotFoundError(
        f"No checkpoint for step {step} was found under {runs_dir}"
    )

  event_run_names = {
      Path(event_file).parent.name
      for event_file in summary.get("event_files", [])
  }
  matching_run = [
      path for path in candidates if path.parent.parent.name in event_run_names
  ]
  if matching_run:
    candidates = matching_run
  return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--summary", type=Path, required=True)
  parser.add_argument("--runs-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--path-only", action="store_true")
  args = parser.parse_args()

  summary = json.loads(args.summary.read_text(encoding="utf-8"))
  selection = select_best(summary)
  checkpoint = find_checkpoint(summary, args.runs_dir, int(selection["step"]))
  report = {**selection, "checkpoint": str(checkpoint.resolve())}

  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
  if args.path_only:
    print(report["checkpoint"])
  else:
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
