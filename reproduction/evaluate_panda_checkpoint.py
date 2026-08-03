"""Independently evaluate a Panda vision checkpoint across held-out seeds."""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any

if sys.platform.startswith("linux"):
  os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from brax.envs.wrappers import training as brax_training  # noqa: E402
from brax.training import checkpoint as brax_checkpoint  # noqa: E402
from brax.training import networks as brax_networks  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import networks_vision  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from ml_collections import config_dict  # noqa: E402
import numpy as np  # noqa: E402

from mujoco_playground import registry  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402
from panda_failure_classification import (  # noqa: E402
    classify_failure,
    summarize_failure_classes,
)


ENV_NAME = "PandaPickCubeCartesian"
APPROACH_DISTANCE_METERS = 0.03
LIFT_HEIGHT_METERS = 0.05
DROP_HEIGHT_METERS = 0.04


class Heartbeat:
  """Periodically reports that a JIT or rollout is still running."""

  def __init__(self, label: str, interval_seconds: float):
    self._label = label
    self._interval_seconds = interval_seconds
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._started = 0.0

  def __enter__(self):
    self._started = time.monotonic()
    if self._interval_seconds > 0:
      self._thread = threading.Thread(target=self._run, daemon=True)
      self._thread.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=1.0)

  def _run(self) -> None:
    while not self._stop.wait(self._interval_seconds):
      elapsed = round(time.monotonic() - self._started)
      print(
          f"[heartbeat] {self._label} | status=running | elapsed={elapsed}s",
          flush=True,
      )


def parse_seeds(value: str) -> list[int]:
  try:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
  except ValueError as error:
    raise argparse.ArgumentTypeError(
        "seeds must be comma-separated integers"
    ) from error
  if not seeds:
    raise argparse.ArgumentTypeError("at least one seed is required")
  if any(seed < 0 for seed in seeds):
    raise argparse.ArgumentTypeError("seeds must be non-negative")
  if len(set(seeds)) != len(seeds):
    raise argparse.ArgumentTypeError("seeds must not contain duplicates")
  return seeds


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError("value must be a positive integer")
  return parsed


def probability(value: str) -> float:
  parsed = float(value)
  if not 0.0 <= parsed <= 1.0:
    raise argparse.ArgumentTypeError("value must be between 0 and 1")
  return parsed


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
  """Returns a two-sided 95% Wilson score interval."""
  if total <= 0:
    raise ValueError("total must be positive")
  z = 1.959963984540054
  rate = successes / total
  denominator = 1.0 + z * z / total
  center = (rate + z * z / (2.0 * total)) / denominator
  margin = (
      z
      * math.sqrt(
          rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
      )
      / denominator
  )
  return (
      max(0.0, min(rate, center - margin)),
      min(1.0, max(rate, center + margin)),
  )


def format_duration(seconds: float) -> str:
  rounded = max(0, round(seconds))
  minutes, seconds = divmod(rounded, 60)
  hours, minutes = divmod(minutes, 60)
  if hours:
    return f"{hours}h{minutes:02d}m{seconds:02d}s"
  return f"{minutes}m{seconds:02d}s"


def print_item(label: str, value: Any) -> None:
  print(f"  {label:<22}{value}", flush=True)


def resolve_checkpoint(path: Path) -> Path:
  checkpoint = path.expanduser().resolve()
  if not checkpoint.is_dir():
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
  if not any(
      (checkpoint / name).is_file()
      for name in ("ppo_network_config.json", "config.json")
  ):
    raise FileNotFoundError(
        "Expected an exact checkpoint directory containing "
        f"ppo_network_config.json or config.json: {checkpoint}"
    )
  return checkpoint


def network_config_path(checkpoint: Path) -> Path:
  for name in ("ppo_network_config.json", "config.json"):
    candidate = checkpoint / name
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(f"Network config not found in {checkpoint}")


def load_network_config(path: Path) -> config_dict.ConfigDict:
  """Loads Brax network config while preserving optional null initializers."""
  loaded = json.loads(path.read_text(encoding="utf-8"))
  observation_size = {}
  for key, serialized_spec in loaded["observation_size"].items():
    shape = (
        serialized_spec["shape"]
        if isinstance(serialized_spec, dict)
        else serialized_spec
    )
    observation_size[key] = tuple(int(dimension) for dimension in shape)
  loaded["observation_size"] = observation_size
  factory = loaded["network_factory_kwargs"]
  activation = factory.get("activation")
  if isinstance(activation, str):
    factory["activation"] = brax_networks.ACTIVATION[activation]
  for name in (
      "policy_network_kernel_init_fn",
      "value_network_kernel_init_fn",
      "q_network_kernel_init_fn",
      "mean_kernel_init_fn",
  ):
    initializer = factory.get(name)
    if isinstance(initializer, str):
      factory[name] = brax_networks.KERNEL_INITIALIZER[initializer]
  return config_dict.create(**loaded)


def git_value(project_dir: Path, *args: str) -> str | None:
  try:
    return subprocess.check_output(
        ["git", *args], cwd=project_dir, text=True, stderr=subprocess.DEVNULL
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def package_versions() -> dict[str, str]:
  packages = (
      "brax",
      "jax",
      "jaxlib",
      "mujoco",
      "mujoco-mjx",
      "playground",
      "warp-lang",
  )
  versions = {}
  for package in packages:
    try:
      versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      versions[package] = "not installed"
  return versions


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description=(
          "Evaluate one exact Panda vision checkpoint across held-out seeds. "
          "Each environment contributes exactly one episode."
      )
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=positive_int, default=256)
  parser.add_argument(
      "--seeds", type=parse_seeds, default=parse_seeds("101,202,303,404")
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument("--target-success", type=probability, default=0.90)
  parser.add_argument("--minimum-seed-success", type=probability, default=0.85)
  parser.add_argument("--box-y-min", type=float)
  parser.add_argument("--box-y-max", type=float)
  parser.add_argument("--contact-capacity-per-env", type=positive_int, default=48)
  parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
  parser.add_argument("--allow-cpu", action="store_true")
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument(
      "--development-episode-length",
      type=positive_int,
      help=argparse.SUPPRESS,
  )
  return parser


def main() -> None:
  run_started = time.monotonic()
  args = build_parser().parse_args()
  project_dir = Path(__file__).resolve().parents[1]
  checkpoint_path = resolve_checkpoint(args.checkpoint)
  if args.heartbeat_seconds < 0:
    raise ValueError("--heartbeat-seconds must be non-negative")
  if (args.box_y_min is None) != (args.box_y_max is None):
    raise ValueError("--box-y-min and --box-y-max must be supplied together")
  if args.box_y_min is not None and args.box_y_min >= args.box_y_max:
    raise ValueError("--box-y-min must be smaller than --box-y-max")

  stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
  output_path = (
      args.output.expanduser().resolve()
      if args.output is not None
      else project_dir
      / "reproduction"
      / "artifacts"
      / "panda-independent-eval"
      / f"evaluation-{stamp}.json"
  )
  if output_path.exists() and not args.overwrite:
    raise FileExistsError(
        f"Output already exists; pass --overwrite to replace it: {output_path}"
    )

  backend = jax.default_backend()
  devices = [str(device) for device in jax.devices()]
  if backend != "gpu" and not args.allow_cpu:
    raise RuntimeError(
        f"JAX backend is {backend}, not gpu. Run this evaluation on the Linux "
        "NVIDIA server, or pass --allow-cpu only for a tiny development test."
    )
  checkpoint_step = (
      int(checkpoint_path.name) if checkpoint_path.name.isdigit() else None
  )
  total_episodes = args.num_envs * len(args.seeds)
  print("[1/4] Runtime and checkpoint validation", flush=True)
  print_item("Backend", backend)
  print_item("Devices", ", ".join(devices))
  print_item(
      "Checkpoint step",
      f"{checkpoint_step:,}" if checkpoint_step is not None else "unknown",
  )
  print_item("Checkpoint", checkpoint_path)

  print("\nEvaluation plan", flush=True)
  print_item("Mode", "deterministic inference only (no training)")
  print_item("Seeds", ", ".join(map(str, args.seeds)))
  print_item("Episodes", f"{len(args.seeds)} x {args.num_envs} = {total_episodes:,}")
  print_item("Aggregate target", f">= {args.target_success:.0%}")
  print_item("Per-seed target", f">= {args.minimum_seed_success:.0%}")
  print_item("Report", output_path)

  if backend == "gpu":
    import warp as wp  # pylint: disable=g-import-not-at-top

    wp.config.log_level = wp.LOG_WARNING

  contact_capacity = args.contact_capacity_per_env * args.num_envs
  print("\n[2/4] Environment and policy loading", flush=True)
  print_item("Vision observation", "64 x 64 RGB")
  print_item("Parallel worlds", args.num_envs)
  print_item("Contact capacity", contact_capacity)
  env_config = registry.get_default_config(ENV_NAME)
  env_overrides = {
      "impl": "warp",
      "vision": True,
      "vision_config.nworld": args.num_envs,
      "naconmax": contact_capacity,
      "naccdmax": contact_capacity,
  }
  base_y_range = [
      -float(env_config.box_init_range),
      float(env_config.box_init_range),
  ]
  if args.box_y_min is not None:
    if args.box_y_min < base_y_range[0] or args.box_y_max > base_y_range[1]:
      raise ValueError(
          f"Requested box y range must stay inside {base_y_range}"
      )
    env_overrides.update({
        "box_init_target_y_range": [args.box_y_min, args.box_y_max],
        "box_init_target_probability": 1.0,
    })
    initial_y_sampling = {
        "mode": "targeted_uniform",
        "range": [args.box_y_min, args.box_y_max],
    }
  else:
    initial_y_sampling = {"mode": "original_uniform", "range": base_y_range}
  print_item(
      "Initial cube y",
      f"{initial_y_sampling['mode']} {initial_y_sampling['range']}",
  )
  base_env = registry.load(
      ENV_NAME, config=env_config, config_overrides=env_overrides
  )
  episode_length = (
      args.development_episode_length or int(env_config.episode_length)
  )
  official_episode_length = episode_length == int(env_config.episode_length)
  if not official_episode_length:
    print(
        "  Warning: development episode length is active; acceptance will "
        "always fail.",
        flush=True,
    )
  wrapped_env = wrapper.wrap_for_brax_training(
      base_env,
      episode_length=episode_length,
      action_repeat=env_config.action_repeat,
  )
  eval_env = brax_training.EvalWrapper(wrapped_env)

  network_config = load_network_config(network_config_path(checkpoint_path))
  ppo_network = brax_checkpoint.get_network(
      network_config, networks_vision.make_ppo_networks_vision
  )
  make_policy = ppo_networks.make_inference_fn(ppo_network)
  params = brax_checkpoint.load(checkpoint_path)
  print_item("Environment", "ready")
  print_item("Policy", "ready")
  print_item("Parameter updates", "disabled")

  reset_fn = jax.jit(eval_env.reset)

  object_body_id = int(
      base_env.unwrapped._obj_body  # pylint: disable=protected-access
  )
  gripper_site_id = int(
      base_env.unwrapped._gripper_site  # pylint: disable=protected-access
  )

  def observe_trajectory(state):
    box_position = state.data.xpos[:, object_body_id]
    gripper_position = state.data.site_xpos[:, gripper_site_id]
    return {
        "box_position": box_position,
        "gripper_box_distance": jnp.linalg.norm(
            box_position - gripper_position, axis=-1
        ),
        "target_height_error": jnp.abs(
            box_position[:, 2] - state.info["target_pos"][:, 2]
        ),
    }

  @jax.jit
  def rollout(state, policy_params, key):
    policy = make_policy(policy_params, deterministic=True)

    initial = observe_trajectory(state)
    batch_size = state.reward.shape[0]
    missing_step = jnp.full((batch_size,), -1, dtype=jnp.int32)
    trajectory = {
        "active": jnp.ones((batch_size,), dtype=bool),
        "min_gripper_box_distance": initial["gripper_box_distance"],
        "min_target_height_error": initial["target_height_error"],
        "max_box_height": initial["box_position"][:, 2],
        "last_active_box_position": initial["box_position"],
        "ever_approached": initial["gripper_box_distance"]
        <= APPROACH_DISTANCE_METERS,
        "ever_reached_box": state.info["reached_box"] > 0.5,
        "ever_lifted": initial["box_position"][:, 2] > LIFT_HEIGHT_METERS,
        "lifted_then_dropped": jnp.zeros((batch_size,), dtype=bool),
        "ever_close_command": jnp.zeros((batch_size,), dtype=bool),
        "ever_hand_box_collision": jnp.zeros((batch_size,), dtype=bool),
        "ever_out_of_bounds_or_invalid": jnp.zeros(
            (batch_size,), dtype=bool
        ),
        "first_approach_step": missing_step,
        "first_reached_box_step": missing_step,
        "first_close_command_step": missing_step,
        "first_lift_step": missing_step,
        "first_success_step": missing_step,
    }

    def first_occurrence(previous, occurred, step_number):
      return jnp.where(
          (previous < 0) & occurred,
          jnp.asarray(step_number, dtype=jnp.int32),
          previous,
      )

    def step(carry, step_index):
      current_state, current_key, current_trajectory = carry
      current_key, action_key = jax.random.split(current_key)
      action, _ = policy(current_state.obs, action_key)
      next_state = eval_env.step(current_state, action)
      active = current_trajectory["active"]
      active_after = active & ~(next_state.done > 0.5)
      observed = observe_trajectory(next_state)

      # AutoReset replaces terminal data with the initial state. Only consume
      # next-state positions for episodes that remain active. Terminal stage
      # flags are read from metrics/info, which retain the true final step.
      distance = jnp.where(
          active_after,
          observed["gripper_box_distance"],
          current_trajectory["min_gripper_box_distance"],
      )
      target_error = jnp.where(
          active_after,
          observed["target_height_error"],
          current_trajectory["min_target_height_error"],
      )
      box_height = jnp.where(
          active_after,
          observed["box_position"][:, 2],
          current_trajectory["max_box_height"],
      )
      approached_now = active & (distance <= APPROACH_DISTANCE_METERS)
      reached_now = active & (next_state.info["reached_box"] > 0.5)
      lifted_now = active & (next_state.metrics["reward/lifted"] > 0.0)
      close_now = active & (action[:, 2] < 0.0)
      collision_now = active & (
          next_state.metrics["reward/no_box_collision"] < 0.5
      )
      invalid_now = active & (
          (next_state.metrics["out_of_bounds"] > 0.5)
          | jnp.isnan(next_state.reward)
      )
      success_now = active & (next_state.metrics["reward/success"] > 0.5)
      ever_lifted = current_trajectory["ever_lifted"] | lifted_now
      dropped_now = (
          active_after
          & ever_lifted
          & (observed["box_position"][:, 2] <= DROP_HEIGHT_METERS)
      )
      step_number = step_index + 1

      updated = {
          "active": active_after,
          "min_gripper_box_distance": jnp.minimum(
              current_trajectory["min_gripper_box_distance"], distance
          ),
          "min_target_height_error": jnp.minimum(
              current_trajectory["min_target_height_error"], target_error
          ),
          "max_box_height": jnp.maximum(
              current_trajectory["max_box_height"], box_height
          ),
          "last_active_box_position": jnp.where(
              active_after[:, None],
              observed["box_position"],
              current_trajectory["last_active_box_position"],
          ),
          "ever_approached": (
              current_trajectory["ever_approached"] | approached_now
          ),
          "ever_reached_box": (
              current_trajectory["ever_reached_box"] | reached_now
          ),
          "ever_lifted": ever_lifted,
          "lifted_then_dropped": (
              current_trajectory["lifted_then_dropped"] | dropped_now
          ),
          "ever_close_command": (
              current_trajectory["ever_close_command"] | close_now
          ),
          "ever_hand_box_collision": (
              current_trajectory["ever_hand_box_collision"] | collision_now
          ),
          "ever_out_of_bounds_or_invalid": (
              current_trajectory["ever_out_of_bounds_or_invalid"]
              | invalid_now
          ),
          "first_approach_step": first_occurrence(
              current_trajectory["first_approach_step"],
              approached_now,
              step_number,
          ),
          "first_reached_box_step": first_occurrence(
              current_trajectory["first_reached_box_step"],
              reached_now,
              step_number,
          ),
          "first_close_command_step": first_occurrence(
              current_trajectory["first_close_command_step"],
              close_now,
              step_number,
          ),
          "first_lift_step": first_occurrence(
              current_trajectory["first_lift_step"],
              lifted_now,
              step_number,
          ),
          "first_success_step": first_occurrence(
              current_trajectory["first_success_step"],
              success_now,
              step_number,
          ),
      }
      return (next_state, current_key, updated), None

    (final_state, _, final_trajectory), _ = jax.lax.scan(
        step,
        (state, key, trajectory),
        jnp.arange(episode_length // env_config.action_repeat),
    )
    return final_state, final_trajectory

  print("\n[3/4] Held-out multi-seed evaluation", flush=True)
  per_seed: list[dict[str, Any]] = []
  episode_records: list[dict[str, Any]] = []
  all_success: list[np.ndarray] = []
  all_rewards: list[np.ndarray] = []
  all_lengths: list[np.ndarray] = []
  object_qpos_address = int(
      base_env.unwrapped._obj_qposadr  # pylint: disable=protected-access
  )
  evaluation_started = time.monotonic()

  for index, seed in enumerate(args.seeds, start=1):
    phase = "JIT compilation + rollout" if index == 1 else "rollout"
    print(f"\nSeed {index}/{len(args.seeds)}: {seed}", flush=True)
    print_item("Phase", phase)
    print_item("Status", "running")
    seed_key = jax.random.PRNGKey(seed)
    reset_key, rollout_key = jax.random.split(seed_key)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    started = time.monotonic()
    with Heartbeat(
        f"seed={seed} phase={phase}", args.heartbeat_seconds
    ):
      initial_state = reset_fn(reset_keys)
      initial_box_positions = np.asarray(
          initial_state.data.qpos[
              :, object_qpos_address : object_qpos_address + 3
          ]
      )
      final_state, trajectory = rollout(initial_state, params, rollout_key)
      eval_metrics = jax.tree.map(
          np.asarray, final_state.info["eval_metrics"]
      )
      trajectory = jax.tree.map(np.asarray, trajectory)

    success_values = np.asarray(
        eval_metrics.episode_metrics["reward/success"]
    ).reshape(-1)
    rewards = np.asarray(eval_metrics.episode_metrics["reward"]).reshape(-1)
    episode_lengths = np.asarray(eval_metrics.episode_steps).reshape(-1)
    success_mask = success_values > 0.5
    successes = int(np.count_nonzero(success_mask))
    duration = time.monotonic() - started
    failure_positions = initial_box_positions[~success_mask]

    for environment_index in range(args.num_envs):
      episode_trajectory = {
          "min_gripper_box_distance": float(
              trajectory["min_gripper_box_distance"][environment_index]
          ),
          "min_target_height_error": float(
              trajectory["min_target_height_error"][environment_index]
          ),
          "max_box_height": float(
              trajectory["max_box_height"][environment_index]
          ),
          "last_active_box_position": np.round(
              trajectory["last_active_box_position"][environment_index], 6
          ).tolist(),
          "ever_approached": bool(
              trajectory["ever_approached"][environment_index]
          ),
          "ever_reached_box": bool(
              trajectory["ever_reached_box"][environment_index]
          ),
          "ever_lifted": bool(
              trajectory["ever_lifted"][environment_index]
          ),
          "lifted_then_dropped": bool(
              trajectory["lifted_then_dropped"][environment_index]
          ),
          "ever_close_command": bool(
              trajectory["ever_close_command"][environment_index]
          ),
          "ever_hand_box_collision": bool(
              trajectory["ever_hand_box_collision"][environment_index]
          ),
          "ever_out_of_bounds_or_invalid": bool(
              trajectory["ever_out_of_bounds_or_invalid"][environment_index]
          ),
          "first_approach_step": int(
              trajectory["first_approach_step"][environment_index]
          ),
          "first_reached_box_step": int(
              trajectory["first_reached_box_step"][environment_index]
          ),
          "first_close_command_step": int(
              trajectory["first_close_command_step"][environment_index]
          ),
          "first_lift_step": int(
              trajectory["first_lift_step"][environment_index]
          ),
          "first_success_step": int(
              trajectory["first_success_step"][environment_index]
          ),
      }
      success = bool(success_mask[environment_index])
      episode_records.append({
          "episode_id": f"{seed}:{environment_index:04d}",
          "seed": seed,
          "environment_index": environment_index,
          "initial_box_position": np.round(
              initial_box_positions[environment_index], 6
          ).tolist(),
          "success": success,
          "success_metric": float(success_values[environment_index]),
          "reward": float(rewards[environment_index]),
          "episode_length": int(episode_lengths[environment_index]),
          "failure_class": (
              None if success else classify_failure(episode_trajectory)
          ),
          "trajectory": episode_trajectory,
      })

    seed_result = {
        "seed": seed,
        "episodes": args.num_envs,
        "successes": successes,
        "failures": args.num_envs - successes,
        "success_rate": successes / args.num_envs,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "duration_seconds": duration,
        "failure_initial_box_positions": np.round(
            failure_positions, 6
        ).tolist(),
    }
    per_seed.append(seed_result)
    all_success.append(success_mask)
    all_rewards.append(rewards)
    all_lengths.append(episode_lengths)
    completed_successes = sum(result["successes"] for result in per_seed)
    completed_episodes = index * args.num_envs
    elapsed_evaluation = time.monotonic() - evaluation_started
    remaining_seeds = len(args.seeds) - index
    estimated_remaining = (
        elapsed_evaluation / index * remaining_seeds
        if remaining_seeds
        else 0.0
    )
    print_item("Status", "complete")
    print_item(
        "Seed success",
        f"{successes}/{args.num_envs} ({seed_result['success_rate']:.2%})",
    )
    print_item(
        "Cumulative success",
        f"{completed_successes}/{completed_episodes} "
        f"({completed_successes / completed_episodes:.2%})",
    )
    print_item("Mean reward", f"{seed_result['mean_reward']:.3f}")
    print_item("Seed elapsed", format_duration(duration))
    print_item("Estimated remaining", f"~{format_duration(estimated_remaining)}")

  success_array = np.concatenate(all_success)
  reward_array = np.concatenate(all_rewards)
  length_array = np.concatenate(all_lengths)
  total = int(success_array.size)
  successes = int(np.count_nonzero(success_array))
  success_rate = successes / total
  interval_low, interval_high = wilson_interval(successes, total)
  worst_seed_rate = min(result["success_rate"] for result in per_seed)
  target_passed = (
      official_episode_length
      and success_rate >= args.target_success
      and worst_seed_rate >= args.minimum_seed_success
  )
  evaluation_duration = time.monotonic() - evaluation_started
  total_duration = time.monotonic() - run_started
  failure_classification = summarize_failure_classes(episode_records)

  git_status = git_value(project_dir, "status", "--porcelain")
  report = {
      "schema_version": 3,
      "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
      "environment": ENV_NAME,
      "checkpoint": str(checkpoint_path),
      "checkpoint_step": checkpoint_step,
      "evaluation": {
          "deterministic_policy": True,
          "held_out_seeds": args.seeds,
          "episodes_per_seed": args.num_envs,
          "episode_length": episode_length,
          "official_episode_length": official_episode_length,
          "contact_capacity": contact_capacity,
          "initial_y_sampling": initial_y_sampling,
          "duration_seconds": evaluation_duration,
          "per_seed": per_seed,
          "episode_records": episode_records,
          "trajectory_diagnostics": {
              "thresholds_meters": {
                  "approach_distance": APPROACH_DISTANCE_METERS,
                  "environment_reached_box_distance": 0.012,
                  "lift_height": LIFT_HEIGHT_METERS,
                  "drop_height": DROP_HEIGHT_METERS,
              },
              "notes": {
                  "last_active_box_position": (
                      "Last non-terminal position because AutoReset replaces "
                      "terminal simulator data."
                  ),
                  "ever_hand_box_collision": (
                      "Collision between the cube and hand capsule; this is "
                      "not interpreted as a successful grasp contact."
                  ),
              },
              "failure_classification": failure_classification,
          },
          "aggregate": {
              "episodes": total,
              "successes": successes,
              "failures": total - successes,
              "success_rate": success_rate,
              "success_rate_wilson_95": [interval_low, interval_high],
              "worst_seed_success_rate": worst_seed_rate,
              "mean_reward": float(np.mean(reward_array)),
              "std_reward": float(np.std(reward_array)),
              "mean_episode_length": float(np.mean(length_array)),
          },
          "acceptance": {
              "target_success_rate": args.target_success,
              "minimum_seed_success_rate": args.minimum_seed_success,
              "passed": target_passed,
          },
      },
      "runtime": {
          "jax_backend": backend,
          "jax_devices": devices,
          "python": platform.python_version(),
          "platform": platform.platform(),
          "packages": package_versions(),
          "git_commit": git_value(project_dir, "rev-parse", "HEAD"),
          "git_branch": git_value(project_dir, "branch", "--show-current"),
          "git_dirty": bool(git_status) if git_status is not None else None,
      },
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = output_path.with_name(f".{output_path.name}.tmp")
  temporary_path.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  temporary_path.replace(output_path)

  status = "PASS" if target_passed else "FAIL"
  seed_summary = " | ".join(
      f"{result['seed']}={result['success_rate']:.2%}" for result in per_seed
  )
  print("\n[4/4] Independent evaluation result")
  print_item("Acceptance", status)
  print_item("Aggregate success", f"{successes}/{total} ({success_rate:.2%})")
  print_item("95% Wilson interval", f"{interval_low:.2%} - {interval_high:.2%}")
  print_item("Worst seed", f"{worst_seed_rate:.2%}")
  print_item("Per-seed success", seed_summary)
  print_item("Mean reward", f"{np.mean(reward_array):.3f}")
  print_item("Recorded failures", total - successes)
  print("\n  Failure classes", flush=True)
  for name, count in failure_classification["counts"].items():
    print(f"    {name:<32}{count:>5}", flush=True)
  print_item("Evaluation time", format_duration(evaluation_duration))
  print_item("Total time", format_duration(total_duration))
  print_item("Report", output_path)


if __name__ == "__main__":
  main()
