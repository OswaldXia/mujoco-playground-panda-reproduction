#!/usr/bin/env python3
"""Audit Markdown math delimiters without requiring a renderer."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "artifacts"}
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")
LATEX_COMMAND_RE = re.compile(
    r"\\(?:begin|end|frac|dfrac|sqrt|gamma|lambda|delta|theta|epsilon|pi|"
    r"hat|text|cdot|qquad|sum|mathbb|operatorname|left|right|exp|log|min|max)\b"
)


@dataclass(frozen=True)
class MathIssue:
  path: Path
  line: int
  message: str

  def display(self, root: Path) -> str:
    return f"{self.path.relative_to(root)}:{self.line}: {self.message}"


def _dollar_positions(text: str) -> list[int]:
  positions: list[int] = []
  index = 0
  while index < len(text):
    if text[index] != "$" or (index > 0 and text[index - 1] == "\\"):
      index += 1
      continue
    if index + 1 < len(text) and text[index + 1] == "$":
      index += 2
      continue
    positions.append(index)
    index += 1
  return positions


def _without_inline_code(text: str) -> tuple[str, list[str]]:
  code_spans: list[str] = []

  def replace(match: re.Match[str]) -> str:
    code_spans.append(match.group(2))
    return " " * len(match.group(0))

  return INLINE_CODE_RE.sub(replace, text), code_spans


def _without_inline_math(text: str, positions: list[int]) -> str:
  chars = list(text)
  for start, end in zip(positions[0::2], positions[1::2]):
    chars[start : end + 1] = " " * (end - start + 1)
  return "".join(chars)


def audit_file(path: Path) -> list[MathIssue]:
  issues: list[MathIssue] = []
  fence: tuple[str, int] | None = None
  in_math_block = False
  math_start = 0
  math_lines: list[str] = []

  for line_number, raw_line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    fence_match = FENCE_RE.match(raw_line)
    if fence_match:
      marker = fence_match.group(1)
      if fence is None:
        fence = (marker[0], len(marker))
      elif marker[0] == fence[0] and len(marker) >= fence[1]:
        fence = None
      continue
    if fence is not None:
      continue

    line, code_spans = _without_inline_code(raw_line)
    for span in code_spans:
      if "$" in span and ("^" in span or "\\" in span):
        issues.append(
            MathIssue(
                path,
                line_number,
                "TeX-like math is inside backticks; use $...$ instead",
            )
        )

    stripped = line.strip()
    if stripped in (r"\[", r"\]", r"\(", r"\)"):
      issues.append(
          MathIssue(
              path,
              line_number,
              "use GitHub-compatible $ or $$ delimiters",
          )
      )
      continue

    if "$$" in line:
      if stripped != "$$":
        issues.append(
            MathIssue(path, line_number, "put $$ on a line by itself")
        )
        continue
      in_math_block = not in_math_block
      if in_math_block:
        math_start = line_number
        math_lines = []
      else:
        expression = "\n".join(math_lines)
        if not expression.strip():
          issues.append(MathIssue(path, math_start, "empty $$ math block"))
        if expression.count("{") != expression.count("}"):
          issues.append(
              MathIssue(path, math_start, "unbalanced braces in $$ math block")
          )
        begins = re.findall(r"\\begin\{([^}]+)\}", expression)
        ends = re.findall(r"\\end\{([^}]+)\}", expression)
        if begins != ends:
          issues.append(
              MathIssue(path, math_start, "unbalanced LaTeX environments")
          )
      continue

    if in_math_block:
      math_lines.append(line)
      continue

    positions = _dollar_positions(line)
    if len(positions) % 2:
      is_currency = (
          len(positions) == 1
          and positions[0] + 1 < len(line)
          and line[positions[0] + 1].isdigit()
      )
      if not is_currency:
        issues.append(
            MathIssue(path, line_number, "unpaired inline $ delimiter")
        )
      continue

    for start, end in zip(positions[0::2], positions[1::2]):
      expression = line[start + 1 : end]
      if not expression:
        issues.append(MathIssue(path, line_number, "empty inline math"))
      elif expression[0].isspace() or expression[-1].isspace():
        issues.append(
            MathIssue(path, line_number, "spaces touch an inline $ delimiter")
        )
      if expression.count("{") != expression.count("}"):
        issues.append(
            MathIssue(path, line_number, "unbalanced braces in inline math")
        )

    outside_math = _without_inline_math(line, positions)
    if LATEX_COMMAND_RE.search(outside_math):
      issues.append(
          MathIssue(path, line_number, "LaTeX command is outside math delimiters")
      )

  if fence is not None:
    # General Markdown fence validation belongs elsewhere; do not duplicate it here.
    pass
  if in_math_block:
    issues.append(MathIssue(path, math_start, "unclosed $$ math block"))
  return issues


def markdown_files(root: Path) -> list[Path]:
  return sorted(
      path
      for path in root.rglob("*.md")
      if not any(part in SKIP_PARTS for part in path.relative_to(root).parts)
  )


def audit_repository(root: Path = ROOT) -> list[MathIssue]:
  return [issue for path in markdown_files(root) for issue in audit_file(path)]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=ROOT)
  args = parser.parse_args()
  root = args.root.resolve()
  files = markdown_files(root)
  issues = audit_repository(root)

  print("\nMarkdown math audit")
  print("=" * 72)
  print(f"  Files checked  {len(files)}")
  if issues:
    for issue in issues:
      print(f"  FAIL  {issue.display(root)}")
    print("-" * 72)
    print(f"  Decision       FAIL ({len(issues)} issue(s))\n")
    return 1
  print("  Syntax         balanced GitHub-compatible delimiters")
  print("-" * 72)
  print("  Decision       PASS\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

