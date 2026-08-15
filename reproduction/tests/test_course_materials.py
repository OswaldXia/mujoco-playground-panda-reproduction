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
    self.assertIn("v0.9", readme)
    self.assertIn("v0.9", status)
    self.assertIn("checkpoint", status)
    self.assertIn("clean-clone", status)


if __name__ == "__main__":
  unittest.main()
