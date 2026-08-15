"""Structural checks for the versioned Panda learning course."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COURSE = ROOT / "docs" / "course"
DOCS = ROOT / "docs"
REQUIRED_SECTIONS = (
    "## 学习目标",
    "## 本章知识清单",
    "## 为什么",
    "## 核心知识",
    "## 最小实验",
    "## 源码定位",
    "## 运行前预测",
    "## 操作",
    "## 预期结果",
    "## 常见错误",
    "## 修改练习",
    "## 自测",
    "## 通过标准",
    "## Git 节点",
)
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


class CourseMaterialsTest(unittest.TestCase):

  def test_course_has_thirteen_complete_modules(self) -> None:
    modules = sorted(COURSE.glob("[0-9][0-9]-*.md"))
    self.assertEqual(
        [path.name[:2] for path in modules], [f"{i:02d}" for i in range(13)]
    )
    for module in modules:
      text = module.read_text(encoding="utf-8")
      missing = [heading for heading in REQUIRED_SECTIONS if heading not in text]
      self.assertFalse(
          missing, f"{module.relative_to(ROOT)} is missing {missing}"
      )

  def test_local_markdown_links_resolve(self) -> None:
    missing: list[str] = []
    for document in DOCS.rglob("*.md"):
      text = document.read_text(encoding="utf-8")
      for raw_target in LINK_PATTERN.findall(text):
        target = raw_target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
          continue
        resolved = (document.parent / target).resolve()
        if not resolved.exists():
          missing.append(f"{document.relative_to(ROOT)} -> {target}")
    self.assertFalse(missing, "broken local links:\n" + "\n".join(missing))

  def test_markdown_math_uses_github_compatible_block_delimiters(self) -> None:
    offenders = []
    for document in COURSE.rglob("*.md"):
      text = document.read_text(encoding="utf-8")
      if any(line in (r"\[", r"\]") for line in text.splitlines()):
        offenders.append(str(document.relative_to(ROOT)))
    self.assertFalse(
        offenders,
        "use $$ blocks instead of \\[ ... \\] for GitHub rendering: "
        + str(offenders),
    )

  def test_course_release_status_is_honest(self) -> None:
    readme = (COURSE / "README.md").read_text(encoding="utf-8")
    status = (COURSE / "CURRICULUM_STATUS.md").read_text(encoding="utf-8")
    self.assertIn("v0.10.1", readme)
    self.assertIn("v0.10.1", status)
    self.assertIn("checkpoint", status)
    self.assertIn("clean-clone", status)

  def test_fresh_environment_entry_materials_exist(self) -> None:
    required = (
        COURSE / "START_HERE.md",
        COURSE / "KNOWLEDGE_MAP.md",
        COURSE / "GATE_RUBRIC.md",
        COURSE / "NOTEBOOK_DESIGN.md",
        DOCS / "labs" / "00_course_preflight.py",
        DOCS / "templates" / "reproduction-contract.md",
        DOCS / "templates" / "source-audit.md",
        DOCS / "templates" / "experiment-record.md",
        DOCS / "templates" / "evaluation-conclusion.md",
    )
    self.assertFalse(
        [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    )

  def test_documented_default_tests_need_no_pytest_extra(self) -> None:
    offenders = []
    for document in DOCS.rglob("*.md"):
      text = document.read_text(encoding="utf-8")
      if "pytest reproduction/tests" in text:
        offenders.append(str(document.relative_to(ROOT)))
    self.assertFalse(offenders, f"pytest-only test commands: {offenders}")

  def test_foundation_micro_lessons_are_complete(self) -> None:
    expected = {
        "01a-vectors-and-frames.md",
        "01b-homogeneous-transforms.md",
        "01c-fk-ik.md",
        "01d-panda-cartesian-control.md",
        "02a-rl-loop.md",
        "02b-return-value-advantage.md",
        "02c-policy-distributions.md",
        "02d-ppo-objective.md",
        "02e-brax-update-shapes.md",
        "04a-functional-jax.md",
        "04b-prng.md",
        "04c-jit-and-tracing.md",
        "04d-vmap-scan-pytree.md",
        "04e-mjx-warp-brax.md",
    }
    foundation_dir = COURSE / "foundations"
    actual = {path.name for path in foundation_dir.glob("*.md")}
    self.assertTrue(expected.issubset(actual), sorted(expected - actual))
    for name in expected:
      text = (foundation_dir / name).read_text(encoding="utf-8")
      self.assertGreater(len(text), 800, f"micro-lesson too short: {name}")
      self.assertIn("通过标准", text, name)

  def test_foundation_reference_solutions_run(self) -> None:
    solution_dir = DOCS / "solutions" / "labs"
    scripts = (
        solution_dir / "01_transform_3d_solution.py",
        solution_dir / "02_advantage_solution.py",
        solution_dir / "04_jax_transforms_solution.py",
    )
    for script in scripts:
      result = subprocess.run(
          [sys.executable, str(script)],
          cwd=ROOT,
          capture_output=True,
          text=True,
          timeout=30,
      )
      self.assertEqual(
          result.returncode,
          0,
          f"{script.relative_to(ROOT)} failed:\n{result.stdout}\n{result.stderr}",
      )
      self.assertIn("PASS", result.stdout)

  def test_offline_gate_four_labs_run_without_gpu(self) -> None:
    scripts = (
        DOCS / "labs" / "09_offline_evaluation.py",
        DOCS / "labs" / "10_offline_failure_analysis.py",
        DOCS / "labs" / "11_offline_integrity_audit.py",
    )
    for script in scripts:
      result = subprocess.run(
          [sys.executable, str(script)],
          cwd=ROOT,
          capture_output=True,
          text=True,
          timeout=10,
      )
      self.assertEqual(
          result.returncode,
          0,
          f"{script.relative_to(ROOT)} failed:\n{result.stdout}\n{result.stderr}",
      )
      self.assertIn("PASS", result.stdout)

  def test_offline_fixture_cannot_be_mistaken_for_rate_sample(self) -> None:
    fixture_path = DOCS / "data" / "guide-free-left-episodes-fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    self.assertEqual(
        fixture["fixture_role"],
        "teaching_only_curated_examples_not_a_rate_sample",
    )
    reference = fixture["formal_reference"]
    self.assertEqual(reference["episodes"], 1024)
    self.assertEqual(reference["successes"], 964)
    self.assertEqual(sum(reference["failure_counts"].values()), 60)
    self.assertEqual(len(fixture["curated_episode_examples"]), 8)

  def test_capstone_requires_compute_matched_control(self) -> None:
    capstone = (COURSE / "12-capstone-experiment.md").read_text(encoding="utf-8")
    rubric = (COURSE / "GATE_RUBRIC.md").read_text(encoding="utf-8")
    for phrase in ("Control", "Treatment", "同一个", "3M", "至少三个"):
      self.assertIn(phrase, capstone)
    self.assertIn("PILOT", rubric)
    self.assertIn("READY", rubric)


if __name__ == "__main__":
  unittest.main()
