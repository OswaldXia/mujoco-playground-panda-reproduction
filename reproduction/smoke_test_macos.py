"""State-environment smoke test for PandaPickCubeCartesian on macOS CPU."""

from __future__ import annotations

import argparse
import importlib.metadata
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


def all_finite(tree: object) -> bool:
  return all(
      bool(np.asarray(jp.all(jp.isfinite(leaf))))
      for leaf in jax.tree.leaves(tree)
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=2)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  config = registry.get_default_config(ENV_NAME)
  env = registry.load(
      ENV_NAME,
      config=config,
      config_overrides={"impl": "jax", "vision": False},
  )

  reset = jax.jit(env.reset)
  step = jax.jit(env.step)

  start = time.monotonic()
  state = reset(jax.random.key(0))
  jax.block_until_ready(state.data.qpos)
  reset_seconds = time.monotonic() - start

  action = jp.zeros(env.action_size)
  step_times = []
  for _ in range(args.steps):
    start = time.monotonic()
    state = step(state, action)
    jax.block_until_ready(state.data.qpos)
    step_times.append(time.monotonic() - start)

  if not all_finite(state.obs):
    raise AssertionError("Observation contains a non-finite value")
  if not bool(np.asarray(jp.isfinite(state.reward))):
    raise AssertionError("Reward is not finite")
  if not bool(np.all(np.isfinite(np.asarray(state.data.qpos)))):
    raise AssertionError("qpos contains a non-finite value")
  if not bool(np.all(np.isfinite(np.asarray(state.data.qvel)))):
    raise AssertionError("qvel contains a non-finite value")

  frames = env.render([state], height=480, width=640)
  image_path = args.output_dir / "panda_state_smoke.png"
  media.write_image(image_path, frames[0])

  report = {
      "environment": ENV_NAME,
      "platform": platform.platform(),
      "python": platform.python_version(),
      "playground": importlib.metadata.version("playground"),
      "jax": importlib.metadata.version("jax"),
      "jax_backend": jax.default_backend(),
      "action_size": env.action_size,
      "qpos_size": int(state.data.qpos.shape[0]),
      "qvel_size": int(state.data.qvel.shape[0]),
      "reward": float(np.asarray(state.reward)),
      "done": float(np.asarray(state.done)),
      "reset_seconds": reset_seconds,
      "step_seconds": step_times,
      "observation_finite": True,
      "state_finite": True,
      "image": str(image_path),
  }
  report_path = args.output_dir / "report.json"
  report_path.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
