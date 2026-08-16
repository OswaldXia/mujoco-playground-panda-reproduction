"""Validate and optionally execute every versioned Panda course notebook."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

try:
  from reproduction.audit_markdown_math import audit_file as audit_math_file
except ModuleNotFoundError:
  from audit_markdown_math import audit_file as audit_math_file


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "notebooks"
NOTEBOOKS = (
    "00_course_dashboard.ipynb",
    "01a_coordinate_representations.ipynb",
    "01b_frames_and_rigid_transforms.ipynb",
    "01_frames_and_transforms.ipynb",
    "02_returns_gae_and_ppo.ipynb",
    "03_mujoco_state_and_control.ipynb",
    "04_jax_execution_model.ipynb",
    "05_panda_environment_dataflow.ipynb",
    "09_evaluation_statistics.ipynb",
    "10_failure_analysis.ipynb",
)
FOUNDATION_NOTEBOOKS = {
    "01a_coordinate_representations.ipynb",
    "01b_frames_and_rigid_transforms.ipynb",
}
REQUIRED_SECTIONS = (
    "## 开始前诊断",
    "## 学习目标",
    "## 本节知识地图",
    "## 关键概念与符号",
    "## 先预测",
    "## 运行与观察",
    "## Worked example",
    "## 故意出错",
    "## 动手修改",
    "## 自测",
    "## 项目源码连接",
    "## Exit ticket",
    "## 学完请记住",
    "## 反思与记录",
)
FOUNDATION_REQUIRED_SECTIONS = (
    "## 严格定义",
    "## 符号、单位与 shape",
    "## 直观解释",
    "## 原理与推导",
    "## 分层练习",
    "## 回看开始诊断",
)
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)")


def load_raw(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def validate_structure(path: Path) -> list[str]:
  errors: list[str] = []
  notebook = load_raw(path)
  if notebook.get("nbformat") != 4:
    errors.append("nbformat must be 4")
  course_meta = notebook.get("metadata", {}).get("course", {})
  for field in ("chapter", "slug", "version", "estimated_minutes"):
    if field not in course_meta:
      errors.append(f"metadata.course.{field} is missing")
  foundation = path.name in FOUNDATION_NOTEBOOKS
  expected_version = "v0.12" if foundation else "v0.11"
  if course_meta.get("version") != expected_version:
    errors.append(f"metadata.course.version must be {expected_version}")
  estimated_minutes = course_meta.get("estimated_minutes")
  maximum_minutes = 150 if foundation else 90
  if (
      not isinstance(estimated_minutes, int)
      or not 1 <= estimated_minutes <= maximum_minutes
  ):
    errors.append(
        f"metadata.course.estimated_minutes must be 1..{maximum_minutes}"
    )
  if foundation and course_meta.get("level") != "foundation":
    errors.append("metadata.course.level must be foundation")

  markdown = "\n".join(
      "".join(cell.get("source", []))
      for cell in notebook.get("cells", [])
      if cell.get("cell_type") == "markdown"
  )
  code = "\n".join(
      "".join(cell.get("source", []))
      for cell in notebook.get("cells", [])
      if cell.get("cell_type") == "code"
  )
  with tempfile.TemporaryDirectory(prefix="panda-course-math-") as temp:
    markdown_path = Path(temp) / f"{path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    for issue in audit_math_file(markdown_path):
      errors.append(f"math line {issue.line}: {issue.message}")
  for section in REQUIRED_SECTIONS:
    if section not in markdown:
      errors.append(f"required section missing: {section}")
  if foundation:
    for section in FOUNDATION_REQUIRED_SECTIONS:
      if section not in markdown:
        errors.append(f"foundation section missing: {section}")
  if "assert_course_kernel" not in code:
    errors.append("kernel guard is missing")
  if "assert " not in code:
    errors.append("no executable self-check assertion found")
  if "check_choice" not in code and "check_value" not in code:
    errors.append("targeted feedback helper is missing")
  if "save_progress" not in code:
    errors.append("local progress hook is missing")
  if foundation:
    comment_lines = sum(
        line.lstrip().startswith("#") for line in code.splitlines()
    )
    if comment_lines < 12:
      errors.append("foundation code needs at least 12 semantic comment lines")
    if code.count("value=None") < 3:
      errors.append("foundation diagnostic choices must default to None")
    if code.count('"exit_answers"'):
      errors.append("exit_answers must be a Python variable, not JSON data")
    exit_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("id", "").endswith("exit-answers")
    ]
    if len(exit_cells) != 1 or exit_cells[0].count("None") < 3:
      errors.append("foundation exit ticket must contain at least 3 blank answers")

  for index, cell in enumerate(notebook.get("cells", [])):
    if not cell.get("id"):
      errors.append(f"cell {index} has no stable nbformat id")
    if cell.get("cell_type") != "code":
      continue
    if cell.get("execution_count") is not None:
      errors.append(f"cell {index} stores an execution count")
    if cell.get("outputs"):
      errors.append(f"cell {index} stores outputs")

  for target in LINK_PATTERN.findall(markdown):
    if "://" in target or target.startswith("mailto:"):
      continue
    if not (path.parent / target).resolve().exists():
      errors.append(f"broken local link: {target}")
  return errors


def _write_temporary_kernel(data_dir: Path) -> None:
  kernel_dir = data_dir / "kernels" / "python3"
  kernel_dir.mkdir(parents=True)
  spec = {
      "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
      "display_name": "Panda Course validation (.venv)",
      "language": "python",
      "metadata": {"debugger": True},
  }
  (kernel_dir / "kernel.json").write_text(
      json.dumps(spec, indent=2) + "\n", encoding="utf-8"
  )


def execute_notebook(path: Path, timeout: int) -> float:
  try:
    import nbformat
    from nbclient import NotebookClient
  except ImportError as exc:
    raise RuntimeError(
        "Notebook execution dependencies are missing. Install .[notebooks]."
    ) from exc

  with tempfile.TemporaryDirectory(prefix="panda-course-notebooks-") as temp:
    temp_path = Path(temp)
    jupyter_data = temp_path / "jupyter-data"
    _write_temporary_kernel(jupyter_data)
    previous = {
        name: os.environ.get(name)
        for name in ("JUPYTER_PATH", "MPLCONFIGDIR", "IPYTHONDIR", "MPLBACKEND")
    }
    os.environ["JUPYTER_PATH"] = str(jupyter_data)
    os.environ["MPLCONFIGDIR"] = str(temp_path / "matplotlib")
    os.environ["IPYTHONDIR"] = str(temp_path / "ipython")
    os.environ["MPLBACKEND"] = "Agg"
    try:
      notebook = nbformat.read(path, as_version=4)
      client = NotebookClient(
          notebook,
          timeout=timeout,
          kernel_name="python3",
          resources={"metadata": {"path": str(ROOT)}},
          allow_errors=False,
      )
      started = time.perf_counter()
      client.execute()
      return time.perf_counter() - started
    finally:
      for name, value in previous.items():
        if value is None:
          os.environ.pop(name, None)
        else:
          os.environ[name] = value


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--structure-only", action="store_true")
  parser.add_argument(
      "--notebook",
      action="append",
      choices=NOTEBOOKS,
      help="Validate only this notebook; repeat to select more than one.",
  )
  parser.add_argument("--timeout", type=int, default=180)
  args = parser.parse_args()

  print("\nPanda course notebook validation")
  print("=" * 76)
  failures = []
  selected_notebooks = tuple(args.notebook) if args.notebook else NOTEBOOKS
  total = len(selected_notebooks)
  for position, name in enumerate(selected_notebooks, start=1):
    path = NOTEBOOK_DIR / name
    if not path.is_file():
      failures.append(f"{name}: file missing")
      print(f"  [{position}/{total}] FAIL  {name}: file missing")
      continue
    errors = validate_structure(path)
    if errors:
      failures.extend(f"{name}: {error}" for error in errors)
      print(f"  [{position}/{total}] FAIL  {name}: {'; '.join(errors)}")
      continue
    if args.structure_only:
      print(f"  [{position}/{total}] PASS  {name}: structure and clean outputs")
      continue
    try:
      seconds = execute_notebook(path, args.timeout)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failures.append(f"{name}: {type(exc).__name__}: {exc}")
      print(f"  [{position}/{total}] FAIL  {name}: {type(exc).__name__}: {exc}")
    else:
      print(f"  [{position}/{total}] PASS  {name}: clean-kernel run in {seconds:.1f}s")

  print("-" * 76)
  if failures:
    print(f"  Decision  FAIL ({len(failures)} issue(s))\n")
    return 1
  mode = "structure" if args.structure_only else "structure + clean-kernel execution"
  print(f"  Decision  PASS ({mode})\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
