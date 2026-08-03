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
from ml_collections import config_dict  # noqa: E402
import numpy as np  # noqa: E402

from mujoco_playground import registry  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402


ENV_NAME = "PandaPickCubeCartesian"


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
  return max(0.0, center - margin), min(1.0, center + margin)


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

  @jax.jit
  def rollout(state, policy_params, key):
    policy = make_policy(policy_params, deterministic=True)

    def step(carry, _):
      current_state, current_key = carry
      current_key, action_key = jax.random.split(current_key)
      action, _ = policy(current_state.obs, action_key)
      next_state = eval_env.step(current_state, action)
      return (next_state, current_key), None

    (final_state, _), _ = jax.lax.scan(
        step,
        (state, key),
        None,
        length=episode_length // env_config.action_repeat,
    )
    return final_state

  print("\n[3/4] Held-out multi-seed evaluation", flush=True)
  per_seed: list[dict[str, Any]] = []
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
      final_state = rollout(initial_state, params, rollout_key)
      eval_metrics = jax.tree.map(
          np.asarray, final_state.info["eval_metrics"]
      )

    success_values = np.asarray(
        eval_metrics.episode_metrics["reward/success"]
    ).reshape(-1)
    rewards = np.asarray(eval_metrics.episode_metrics["reward"]).reshape(-1)
    episode_lengths = np.asarray(eval_metrics.episode_steps).reshape(-1)
    success_mask = success_values > 0.5
    successes = int(np.count_nonzero(success_mask))
    duration = time.monotonic() - started
    failure_positions = initial_box_positions[~success_mask]

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

  git_status = git_value(project_dir, "status", "--porcelain")
  report = {
      "schema_version": 1,
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
          "duration_seconds": evaluation_duration,
          "per_seed": per_seed,
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
  print_item("Evaluation time", format_duration(evaluation_duration))
  print_item("Total time", format_duration(total_duration))
  print_item("Report", output_path)


if __name__ == "__main__":
  main()
