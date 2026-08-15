"""Structural checks for the versioned Panda learning course."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COURSE = ROOT / "docs" / "course"
DOCS = ROOT / "docs"
REQUIRED_SECTIONS = (
    "## 学习目标",
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

  def test_course_release_status_is_honest(self) -> None:
    readme = (COURSE / "README.md").read_text(encoding="utf-8")
    status = (COURSE / "CURRICULUM_STATUS.md").read_text(encoding="utf-8")
    self.assertIn("v0.9.1", readme)
    self.assertIn("v0.9.1", status)
    self.assertIn("checkpoint", status)
    self.assertIn("clean-clone", status)

  def test_fresh_environment_entry_materials_exist(self) -> None:
    required = (
        COURSE / "START_HERE.md",
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


if __name__ == "__main__":
  unittest.main()
