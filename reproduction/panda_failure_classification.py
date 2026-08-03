"""Classify Panda rollout failures from trajectory-level diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


FAILURE_CLASS_ORDER = (
    "out_of_bounds_or_invalid",
    "never_approached",
    "approached_not_reached",
    "reached_no_lift",
    "lifted_then_dropped",
    "lifted_timeout",
)

FAILURE_CLASS_DESCRIPTIONS = {
    "out_of_bounds_or_invalid": (
        "Cube left the valid workspace or the simulation became invalid."
    ),
    "never_approached": (
        "Gripper never came within the diagnostic approach threshold."
    ),
    "approached_not_reached": (
        "Gripper approached the cube but never met the environment's "
        "12 mm reached-box condition."
    ),
    "reached_no_lift": (
        "The reached-box condition occurred, but the cube was never lifted."
    ),
    "lifted_then_dropped": (
        "The cube was lifted and later fell below the drop threshold."
    ),
    "lifted_timeout": (
        "The cube was lifted but the success condition was not reached before "
        "termination."
    ),
}


def classify_failure(trajectory: Mapping[str, Any]) -> str:
  """Returns one mutually exclusive failure class for a failed episode."""
  if bool(trajectory.get("ever_out_of_bounds_or_invalid", False)):
    return "out_of_bounds_or_invalid"
  if bool(trajectory.get("ever_lifted", False)):
    if bool(trajectory.get("lifted_then_dropped", False)):
      return "lifted_then_dropped"
    return "lifted_timeout"
  if not bool(trajectory.get("ever_approached", False)):
    return "never_approached"
  if not bool(trajectory.get("ever_reached_box", False)):
    return "approached_not_reached"
  return "reached_no_lift"


def summarize_failure_classes(
    episode_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Builds stable counts and representative episode identifiers."""
  failures = [record for record in episode_records if not record["success"]]
  counts = Counter(str(record["failure_class"]) for record in failures)
  unexpected = sorted(set(counts).difference(FAILURE_CLASS_ORDER))
  if unexpected:
    raise ValueError(f"Unknown failure classes: {unexpected}")

  ordered_counts = {
      name: counts.get(name, 0) for name in FAILURE_CLASS_ORDER
  }
  representatives: dict[str, list[str]] = {}
  for name in FAILURE_CLASS_ORDER:
    representatives[name] = [
        str(record["episode_id"])
        for record in failures
        if record["failure_class"] == name
    ][:5]

  total = len(failures)
  dominant = max(ordered_counts, key=ordered_counts.get) if total else None
  return {
      "total_failures": total,
      "counts": ordered_counts,
      "shares": {
          name: count / total if total else 0.0
          for name, count in ordered_counts.items()
      },
      "dominant_class": dominant,
      "descriptions": FAILURE_CLASS_DESCRIPTIONS,
      "representative_episode_ids": representatives,
  }
