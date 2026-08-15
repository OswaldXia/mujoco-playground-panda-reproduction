"""Complete vmap, scan, and PRNG-key exercises.

Hints:
  Level 1: jax.vmap(single_step) creates the environment batch dimension.
  Level 2: lax.scan body returns (new_carry, output_for_this_step).
  Level 3: split one root key into batch_size independent sample keys.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def batched_step(position: jax.Array, velocity: jax.Array) -> jax.Array:
  """Applies position + 0.1 * velocity to every batch row."""
  del position, velocity
  raise NotImplementedError("TODO: implement batched_step with vmap")


def rollout(initial: jax.Array, actions: jax.Array) -> jax.Array:
  """Returns positions after each action using lax.scan."""
  del initial, actions
  raise NotImplementedError("TODO: implement rollout with lax.scan")


def independent_uniform(seed: int, batch_size: int) -> jax.Array:
  """Returns one scalar sample from each independently split key."""
  del seed, batch_size
  raise NotImplementedError("TODO: split and vmap random.uniform")


def main() -> None:
  positions = jnp.zeros((4, 2))
  velocities = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
  stepped = batched_step(positions, velocities)
  assert stepped.shape == (4, 2)
  assert bool(jnp.allclose(stepped, velocities * 0.1))

  actions = jnp.ones((3, 2))
  trajectory = rollout(jnp.zeros(2), actions)
  assert trajectory.shape == (3, 2)
  assert bool(jnp.allclose(trajectory[-1], jnp.array([3.0, 3.0])))

  samples = independent_uniform(7, 4)
  assert samples.shape == (4,)
  assert len(set(map(float, samples))) == 4
  print("PASS: vmap, scan, and split keys have the predicted shapes")


if __name__ == "__main__":
  main()
