"""Small, dependency-light feedback and progress helpers for course notebooks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VALID_LEVELS = ("red", "yellow", "green")
LEVEL_LABELS = {
    "red": "🔴 未建立概念",
    "yellow": "🟡 能解释但不会独立使用",
    "green": "🟢 能解释、验证并迁移",
}


def check_value(
    label: str,
    actual: Any,
    expected: Any,
    *,
    hint: str,
    atol: float = 1e-8,
) -> bool:
  """Checks a numerical answer and prints actionable learner feedback."""
  try:
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=0.0))
  except (TypeError, ValueError):
    passed = actual == expected
  if passed:
    print(f"[PASS] {label}: your prediction matches the evidence.")
  else:
    print(f"[NEXT] {label}: expected {expected!r}, received {actual!r}.")
    print(f"       Hint: {hint}")
  return passed


def check_choice(
    label: str,
    answer: str,
    expected: str,
    *,
    hint: str,
    explanation: str,
) -> bool:
  """Checks a short conceptual answer without hiding the explanation."""
  normalized = answer.strip().casefold()
  passed = normalized == expected.strip().casefold()
  if passed:
    print(f"[PASS] {label}: {explanation}")
  else:
    print(f"[NEXT] {label}: '{answer}' is not the intended choice.")
    print(f"       Hint: {hint}")
  return passed


def progress_file(root: Path) -> Path:
  return root / "reproduction" / "artifacts" / "course-progress" / "progress.json"


def load_progress(root: Path) -> dict[str, Any]:
  path = progress_file(root)
  if not path.is_file():
    return {"schema_version": 1, "notebooks": {}}
  data = json.loads(path.read_text(encoding="utf-8"))
  if data.get("schema_version") != 1 or not isinstance(data.get("notebooks"), dict):
    raise ValueError(f"Unsupported course progress file: {path}")
  return data


def save_progress(
    root: Path,
    notebook: str,
    concepts: dict[str, str],
    *,
    exit_ticket_passed: bool,
) -> Path:
  """Stores local learning state outside Git-tracked course materials."""
  invalid = {level for level in concepts.values() if level not in VALID_LEVELS}
  if invalid:
    raise ValueError(f"Unknown progress levels: {sorted(invalid)}")
  data = load_progress(root)
  data["notebooks"][notebook] = {
      "concepts": concepts,
      "exit_ticket_passed": bool(exit_ticket_passed),
      "updated_at": datetime.now(UTC).isoformat(),
  }
  path = progress_file(root)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  return path


def progress_rows(
    progress: dict[str, Any], order: Iterable[tuple[str, str]]
) -> list[tuple[str, str, str]]:
  """Returns dashboard rows without requiring IPython display machinery."""
  notebooks = progress.get("notebooks", {})
  rows = []
  for slug, title in order:
    record = notebooks.get(slug)
    if not record:
      rows.append((title, "未开始", "—"))
      continue
    levels = list(record.get("concepts", {}).values())
    green = sum(level == "green" for level in levels)
    status = f"{green}/{len(levels)} 个知识点为绿" if levels else "未自评"
    exit_status = "PASS" if record.get("exit_ticket_passed") else "未通过"
    rows.append((title, status, exit_status))
  return rows


def format_progress_table(rows: Iterable[tuple[str, str, str]]) -> str:
  lines = [
      "| Notebook | 知识点状态 | Exit ticket |",
      "| --- | --- | --- |",
  ]
  lines.extend(f"| {title} | {status} | {exit_status} |" for title, status, exit_status in rows)
  return "\n".join(lines)

