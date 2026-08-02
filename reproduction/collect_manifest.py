"""Collect a compact, machine-readable reproduction environment manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


PACKAGES = (
    "playground",
    "jax",
    "jaxlib",
    "mujoco",
    "mujoco-mjx",
    "warp-lang",
    "brax",
    "flax",
    "orbax-checkpoint",
)


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], check=True, capture_output=True, text=True
  )
  return result.stdout.strip()


def package_versions() -> dict[str, str]:
  versions = {}
  for package in PACKAGES:
    try:
      versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      versions[package] = "not-installed"
  return versions


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  manifest: dict[str, object] = {
      "platform": platform.platform(),
      "machine": platform.machine(),
      "python": platform.python_version(),
      "git_commit": git_output("rev-parse", "HEAD"),
      "git_branch": git_output("branch", "--show-current"),
      "git_dirty": bool(git_output("status", "--porcelain")),
      "packages": package_versions(),
  }

  try:
    import jax

    manifest["jax_backend"] = jax.default_backend()
    manifest["jax_devices"] = [str(device) for device in jax.devices()]
  except Exception as exc:  # pylint: disable=broad-exception-caught
    manifest["jax_error"] = repr(exc)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
