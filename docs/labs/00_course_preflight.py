"""Zero-dependency readiness check for the Panda learning course."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE_PACKAGES = (
    "jax",
    "jaxlib",
    "mujoco",
    "mujoco-mjx",
    "brax",
    "warp-lang",
    "playground",
)
NOTEBOOK_PACKAGES = ("jupyter", "ipykernel", "nbclient", "nbformat", "matplotlib")


def _git(*args: str) -> str | None:
  try:
    result = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
  except (FileNotFoundError, subprocess.SubprocessError):
    return None
  return result.stdout.strip()


def _package_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
  versions: dict[str, str | None] = {}
  for package in packages:
    try:
      versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      versions[package] = None
  return versions


def _find_checkpoints() -> int:
  artifacts = ROOT / "reproduction" / "artifacts"
  if not artifacts.exists():
    return 0
  return sum(
      1
      for path in artifacts.glob("**/checkpoints/*")
      if path.is_dir() and path.name.isdigit()
  )


def collect() -> dict[str, object]:
  versions = _package_versions(CORE_PACKAGES)
  notebook_versions = _package_versions(NOTEBOOK_PACKAGES)
  branch = _git("branch", "--show-current")
  dirty = _git("status", "--porcelain")
  in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
  return {
      "repository_root": str(ROOT),
      "repository_layout_ok": (ROOT / "pyproject.toml").is_file()
      and (ROOT / "docs" / "course" / "README.md").is_file(),
      "python": platform.python_version(),
      "python_supported": sys.version_info[:2] in {(3, 11), (3, 12), (3, 13)},
      "virtual_environment": in_venv,
      "virtual_environment_path": sys.prefix if in_venv else None,
      "platform": platform.platform(),
      "machine": platform.machine(),
      "git_available": shutil.which("git") is not None,
      "git_branch": branch,
      "git_dirty": bool(dirty) if dirty is not None else None,
      "ripgrep_available": shutil.which("rg") is not None,
      "packages": versions,
      "core_packages_complete": all(versions.values()),
      "notebook_packages": notebook_versions,
      "notebook_packages_complete": all(notebook_versions.values()),
      "notebook_files": len(list((ROOT / "docs" / "notebooks").glob("*.ipynb"))),
      "checkpoint_directories": _find_checkpoints(),
      "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
  }


def _line(label: str, status: str, detail: str) -> None:
  print(f"  [{status:<4}] {label:<22} {detail}")


def print_report(report: dict[str, object]) -> None:
  print("\nPanda course preflight")
  print("=" * 72)
  _line(
      "Repository",
      "PASS" if report["repository_layout_ok"] else "FAIL",
      str(report["repository_root"]),
  )
  _line(
      "Python",
      "PASS" if report["python_supported"] else "WARN",
      f"{report['python']} (course-tested: 3.11-3.13)",
  )
  _line(
      "Virtual environment",
      "PASS" if report["virtual_environment"] else "WARN",
      str(report["virtual_environment_path"] or "not active"),
  )
  _line(
      "Git branch",
      "PASS" if report["git_available"] else "WARN",
      str(report["git_branch"] or "unavailable"),
  )
  dirty_detail = (
      "working tree has changes" if report["git_dirty"] else "working tree clean"
  )
  _line("Git status", "WARN" if report["git_dirty"] else "PASS", dirty_detail)
  _line(
      "Source search",
      "PASS" if report["ripgrep_available"] else "WARN",
      "rg" if report["ripgrep_available"] else "use grep -RIn fallback",
  )

  print("\nCore packages")
  for name, version in dict(report["packages"]).items():
    _line(name, "PASS" if version else "WARN", str(version or "not installed"))

  print("\nInteractive notebooks")
  for name, version in dict(report["notebook_packages"]).items():
    _line(name, "PASS" if version else "WARN", str(version or "not installed"))
  notebook_ready = bool(
      report["virtual_environment"]
      and report["notebook_packages_complete"]
      and report["notebook_files"] == 5
  )
  _line(
      "Notebook course",
      "PASS" if notebook_ready else "WARN",
      f"{report['notebook_files']}/5 files; "
      + ("project kernel ready" if notebook_ready else "install .[notebooks]"),
  )

  print("\nAvailable learning tracks")
  _line(
      "A / Mac foundations",
      "PASS" if notebook_ready else "WARN",
      "chapters 00-06, notebooks, and source labs",
  )
  _line("B / Offline analysis", "PASS", "bundled fixtures; no GPU required")
  gpu_ready = bool(report["nvidia_smi_available"] and report["core_packages_complete"])
  _line(
      "C / Linux GPU",
      "PASS" if gpu_ready else "WARN",
      "candidate ready" if gpu_ready else "requires Linux NVIDIA setup",
  )
  _line(
      "Local checkpoints",
      "PASS" if report["checkpoint_directories"] else "WARN",
      f"{report['checkpoint_directories']} checkpoint directories found",
  )

  print("\nNext action")
  if not report["repository_layout_ok"]:
    print("  [NEXT] Enter the repository root and run this script again.")
  elif not report["python_supported"]:
    print("  [NEXT] Create a Python 3.12 virtual environment; see chapter 06.")
  elif not report["virtual_environment"] or not report["core_packages_complete"]:
    print("  [NEXT] Continue chapter 00 now; install the environment before chapter 03.")
  else:
    print("  [NEXT] Open docs/course/00-project-map.md and start your contract.")
  print()


def main() -> int:
  report = collect()
  if "--json" in sys.argv[1:]:
    print(json.dumps(report, indent=2, sort_keys=True))
  else:
    print_report(report)
  return 0 if report["repository_layout_ok"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
