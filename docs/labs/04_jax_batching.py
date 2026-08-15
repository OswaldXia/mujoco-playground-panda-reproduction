"""Demonstrate explicit PRNG keys, vmap, and JIT timing."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp


def energy(x: jax.Array) -> jax.Array:
  return jnp.sum(x * x)


def timed_call(fn, value: jax.Array) -> tuple[jax.Array, float]:
  start = time.perf_counter()
  result = fn(value)
  jax.block_until_ready(result)
  return result, time.perf_counter() - start


def main() -> None:
  batch = jnp.arange(24, dtype=jnp.float32).reshape(8, 3)
  vectorized = jax.vmap(energy)
  compiled = jax.jit(vectorized)

  expected = jnp.stack([energy(row) for row in batch])
  actual = vectorized(batch)
  if not bool(jnp.allclose(expected, actual)):
    raise AssertionError("vmap does not match the explicit map")

  _, first_seconds = timed_call(compiled, batch)
  _, cached_seconds = timed_call(compiled, batch)

  key = jax.random.key(7)
  repeated_a = jax.random.uniform(key, (3,))
  repeated_b = jax.random.uniform(key, (3,))
  key_a, key_b = jax.random.split(key)
  split_a = jax.random.uniform(key_a, (3,))
  split_b = jax.random.uniform(key_b, (3,))

  print(f"backend={jax.default_backend()} devices={jax.devices()}")
  print(f"batch_shape={batch.shape} output_shape={actual.shape}")
  print(f"first_jit_seconds={first_seconds:.6f} cached_seconds={cached_seconds:.6f}")
  print(f"same_key_repeats={bool(jnp.array_equal(repeated_a, repeated_b))}")
  print(f"split_keys_differ={not bool(jnp.array_equal(split_a, split_b))}")


if __name__ == "__main__":
  main()
