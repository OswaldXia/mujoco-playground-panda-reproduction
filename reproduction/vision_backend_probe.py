"""Probe the Panda MJWarp RGB path and write a structured capability report."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy as media
import numpy as np

from mujoco_playground import registry


ENV_NAME = "PandaPickCubeCartesian"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--image", type=Path)
  parser.add_argument("--require-ready", action="store_true")
  parser.add_argument("--require-gpu", action="store_true")
  args = parser.parse_args()

  report: dict[str, object] = {
      "environment": ENV_NAME,
      "platform": platform.platform(),
      "jax_backend": jax.default_backend(),
      "jax_devices": [str(device) for device in jax.devices()],
      "ready": False,
  }

  try:
    env = registry.load(
        ENV_NAME,
        config=registry.get_default_config(ENV_NAME),
        config_overrides={
            "impl": "warp",
            "vision": True,
            "vision_config.nworld": 1,
            "naconmax": 24,
            "naccdmax": 24,
        },
    )
    reset = jax.jit(env.reset)
    start = time.monotonic()
    state = reset(jax.random.key(0))
    rgb = state.obs["pixels/view_0"]
    jax.block_until_ready(rgb)
    reset_seconds = time.monotonic() - start

    step = jax.jit(env.step)
    start = time.monotonic()
    state = step(state, jp.zeros(env.action_size))
    rgb = state.obs["pixels/view_0"]
    jax.block_until_ready(rgb)
    step_seconds = time.monotonic() - start
    rgb_array = np.asarray(rgb)
    if args.image is not None:
      args.image.parent.mkdir(parents=True, exist_ok=True)
      image = rgb_array[0] if rgb_array.ndim == 4 else rgb_array
      media.write_image(args.image, image)
    report.update({
        "ready": True,
        "reset_seconds": reset_seconds,
        "step_seconds": step_seconds,
        "rgb_shape": list(rgb_array.shape),
        "rgb_dtype": str(rgb_array.dtype),
        "rgb_min": float(rgb_array.min()),
        "rgb_max": float(rgb_array.max()),
        "rgb_finite": bool(np.isfinite(rgb_array).all()),
        "image": str(args.image) if args.image is not None else None,
    })
  except Exception as exc:  # pylint: disable=broad-exception-caught
    report.update({
        "error_type": type(exc).__name__,
        "error": str(exc),
    })

  report["gpu_training_ready"] = bool(
      report["ready"] and report["jax_backend"] == "gpu"
  )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, indent=2, sort_keys=True))

  if args.require_ready and not report["ready"]:
    raise SystemExit(1)
  if args.require_gpu and not report["gpu_training_ready"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
