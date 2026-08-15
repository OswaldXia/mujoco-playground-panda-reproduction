"""Reference solution for vmap, scan, and PRNG keys."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def batched_step(position: jax.Array, velocity: jax.Array) -> jax.Array:
  def single_step(pos, vel):
    return pos + 0.1 * vel

  return jax.vmap(single_step)(position, velocity)


def rollout(initial: jax.Array, actions: jax.Array) -> jax.Array:
  def body(position, action):
    next_position = position + action
    return next_position, next_position

  _, trajectory = jax.lax.scan(body, initial, actions)
  return trajectory


def independent_uniform(seed: int, batch_size: int) -> jax.Array:
  keys = jax.random.split(jax.random.key(seed), batch_size)
  return jax.vmap(lambda key: jax.random.uniform(key, ()))(keys)


def main() -> None:
  positions = jnp.zeros((4, 2))
  velocities = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
  assert bool(jnp.allclose(batched_step(positions, velocities), velocities * 0.1))
  trajectory = rollout(jnp.zeros(2), jnp.ones((3, 2)))
  assert trajectory.shape == (3, 2)
  assert bool(jnp.allclose(trajectory[-1], jnp.array([3.0, 3.0])))
  samples = independent_uniform(7, 4)
  assert samples.shape == (4,)
  assert len(set(map(float, samples))) == 4
  print("PASS: JAX transforms solution")


if __name__ == "__main__":
  main()
