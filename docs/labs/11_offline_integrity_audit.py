"""Compare historical guide-assisted and guide-free evaluation protocols."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "reproduction" / "results" / "linux-trajectory-failure-analysis.json"
FORMAL = (
    ROOT / "reproduction" / "results" / "linux-guide-free-left-trajectory-analysis.json"
)


def main() -> None:
  historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
  formal = json.loads(FORMAL.read_text(encoding="utf-8"))
  old_protocol = historical["protocol"]
  integrity = historical["evaluation_integrity_finding"]
  new_protocol = formal["protocol"]

  checks = {
      "historical_is_labeled": old_protocol["status"] == "historical_guide_assisted",
      "historical_guide_was_nonzero": integrity["configured_guide_swap_probability"] > 0,
      "historical_step1_anomaly_exists": integrity["step_1_reached_episodes"] > 0,
      "formal_is_guide_free": new_protocol["guide_swap_probability"] == 0.0,
      "formal_schema_is_newer": new_protocol["report_schema_version"]
      > old_protocol["report_schema_version"],
  }
  if not all(checks.values()):
    raise ValueError(f"integrity checks failed: {checks}")

  print("\nOffline evaluation-integrity audit")
  print("=" * 76)
  print("  Protocol                 Historical             Formal")
  print("-" * 76)
  print(f"  Status                   {old_protocol['status']:<22} {new_protocol['status']}")
  print(
      f"  Report schema            {old_protocol['report_schema_version']:<22} "
      f"{new_protocol['report_schema_version']}"
  )
  print(
      f"  Guide probability        {integrity['configured_guide_swap_probability']:<22} "
      f"{new_protocol['guide_swap_probability']}"
  )
  print(f"  Step-1 reached anomaly   {integrity['step_1_reached_episodes']:<22} not present by protocol")
  print("-" * 76)
  print("  Decision                 DO NOT pool or subtract rows across protocols")
  print("\nPASS: historical evidence is retained but excluded from formal acceptance\n")


if __name__ == "__main__":
  main()
