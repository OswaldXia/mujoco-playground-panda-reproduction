"""Tests for opt-in targeted Panda initial-position sampling."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mujoco_playground._src.manipulation.franka_emika_panda import pick_cartesian


class PandaPositionSamplingTest(unittest.TestCase):

  def test_guide_swap_default_preserves_training_exploration(self):
    probability = pick_cartesian.default_config().guide_swap_probability
    self.assertEqual(probability, 0.05)
    keys = jax.random.split(jax.random.PRNGKey(2), 512)
    expected = jax.vmap(lambda key: jax.random.bernoulli(key, 0.05))(keys)
    actual = jax.vmap(
        lambda key: pick_cartesian.sample_guide_swap(
            key, jnp.array(True), probability
        )
    )(keys)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

  def test_guide_swap_probability_zero_disables_evaluation_aid(self):
    keys = jax.random.split(jax.random.PRNGKey(3), 256)
    swaps = jax.vmap(
        lambda key: pick_cartesian.sample_guide_swap(
            key, jnp.array(True), 0.0
        )
    )(keys)
    self.assertFalse(bool(jnp.any(swaps)))

  def test_guide_swap_only_applies_on_new_reset(self):
    key = jax.random.PRNGKey(5)
    self.assertTrue(
        bool(pick_cartesian.sample_guide_swap(key, jnp.array(True), 1.0))
    )
    self.assertFalse(
        bool(pick_cartesian.sample_guide_swap(key, jnp.array(False), 1.0))
    )

  def test_default_sampling_preserves_original_rng_path(self):
    key = jax.random.PRNGKey(7)
    expected = jax.random.uniform(key, (), minval=-0.05, maxval=0.05)
    actual = pick_cartesian.sample_box_y(key, 0.05, (-0.05, -0.02), 0.0)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

  def test_target_probability_one_stays_inside_target_range(self):
    keys = jax.random.split(jax.random.PRNGKey(11), 2048)
    values = jax.vmap(
        lambda key: pick_cartesian.sample_box_y(
            key, 0.05, (-0.05, -0.02), 1.0
        )
    )(keys)
    self.assertTrue(bool(jnp.all(values >= -0.05)))
    self.assertTrue(bool(jnp.all(values < -0.02)))

  def test_half_mixture_biases_mean_left_without_losing_base_support(self):
    keys = jax.random.split(jax.random.PRNGKey(13), 8192)
    values = jax.vmap(
        lambda key: pick_cartesian.sample_box_y(
            key, 0.05, (-0.05, -0.02), 0.5
        )
    )(keys)
    values = np.asarray(values)
    self.assertGreater(float(values.mean()), -0.020)
    self.assertLess(float(values.mean()), -0.015)
    self.assertTrue(np.any(values > 0.03))
    self.assertTrue(np.all(values >= -0.05))
    self.assertTrue(np.all(values < 0.05))


if __name__ == "__main__":
  unittest.main()
