"""Load the state Panda task and print its learning contract."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mujoco_playground import registry


ENV_NAME = "PandaPickCubeCartesian"


def main() -> None:
  config = registry.get_default_config(ENV_NAME)
  env = registry.load(
      ENV_NAME,
      config=config,
      config_overrides={"impl": "jax", "vision": False},
  )
  state = jax.jit(env.reset)(jax.random.key(0))
  jax.block_until_ready(state.data.qpos)
  next_state = jax.jit(env.step)(state, jnp.zeros(env.action_size))
  jax.block_until_ready(next_state.data.qpos)

  box_qpos_address = env._obj_qposadr  # pylint: disable=protected-access
  box_position = np.asarray(
      state.data.qpos[box_qpos_address : box_qpos_address + 3]
  )
  print(f"environment={ENV_NAME}")
  print(f"backend={jax.default_backend()} action_size={env.action_size}")
  print(f"observation_shape={np.asarray(state.obs).shape}")
  print(f"box_position={box_position.tolist()}")
  print(f"ctrl_dt={config.ctrl_dt} sim_dt={config.sim_dt}")
  print(f"episode_length={config.episode_length}")
  print(f"box_init_range={config.box_init_range}")
  print(f"guide_swap_probability={config.guide_swap_probability}")
  print(f"zero_action_reward={float(np.asarray(next_state.reward)):.6f}")
  print(f"zero_action_done={float(np.asarray(next_state.done)):.1f}")


if __name__ == "__main__":
  main()
