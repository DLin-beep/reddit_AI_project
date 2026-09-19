"""Planted geometric checks for frozen kNN level sets and heldout rejection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_region_density import (  # noqa: E402
    density_level_set_summaries,
    fit_density_regions,
    predict_density_regions,
)


def circle(angles: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(angles), np.sin(angles)))


class DensityRegionTests(unittest.TestCase):
    def test_two_separated_groups_and_distant_rejection(self) -> None:
        data = circle(np.r_[np.linspace(-0.04, 0.04, 12), np.pi + np.linspace(-0.04, 0.04, 12)])
        model = fit_density_regions(data, n_neighbors=3, density_quantile=0, min_cluster_size=4)
        self.assertEqual(model["component_sizes"], [12, 12])
        self.assertEqual(len(set(model["labels"][:12])), 1)
        self.assertNotEqual(model["labels"][0], model["labels"][12])
        prediction = predict_density_regions(circle(np.array([0, np.pi, np.pi / 2])), model)
        np.testing.assert_array_equal(prediction, [model["labels"][0], model["labels"][12], -1])

    def test_retention_and_regions_are_nested_with_frozen_graph(self) -> None:
        rng = np.random.default_rng(42)
        data = circle(np.r_[rng.normal(0, 0.03, 70), rng.normal(1.5, 0.12, 50), rng.uniform(-np.pi, np.pi, 20)])
        models = [fit_density_regions(data, 5, quantile, 3) for quantile in (0, 0.5, 0.8)]
        for lower, higher in zip(models[:-1], models[1:]):
            self.assertTrue(np.all(~higher["retained_mask"] | lower["retained_mask"]))
            for region in range(higher["n_regions"]):
                parents = np.unique(lower["labels"][higher["labels"] == region])
                self.assertEqual(len(parents), 1)
                self.assertGreaterEqual(parents[0], 0)
        summaries = density_level_set_summaries(models[0])
        self.assertEqual([entry["n_retained"] for entry in summaries], [entry["n_retained"] for entry in models])
        np.testing.assert_array_equal(models[0]["labels"], fit_density_regions(data, 5, 0, 3)["labels"])

    def test_small_components_remain_noise(self) -> None:
        data = circle(np.r_[np.linspace(-0.1, 0.1, 10), [2, 2.001, 2.002]])
        model = fit_density_regions(data, 2, 0, 4)
        self.assertEqual(model["n_regions"], 1)
        np.testing.assert_array_equal(model["labels"][-3:], [-1, -1, -1])
        self.assertEqual(predict_density_regions(circle(np.array([2.0005])), model)[0], -1)

    def test_all_noise_and_empty_prediction(self) -> None:
        data = circle(np.linspace(0, 0.05, 5))
        model = fit_density_regions(data, 2, 0.8, 10)
        self.assertEqual(model["n_regions"], 0)
        np.testing.assert_array_equal(predict_density_regions(data, model), np.full(5, -1))
        self.assertEqual(predict_density_regions(np.empty((0, 2)), model).shape, (0,))

    def test_duplicate_points_exclude_self_and_allow_exact_matches(self) -> None:
        data = np.array([[1.0, 0.0]] * 4 + [[-1.0, 0.0]] * 4, dtype=np.float32)
        model = fit_density_regions(data, 2, 0.8, 3)
        self.assertEqual(model["n_regions"], 2)
        self.assertEqual(model["n_retained"], 8)  # Quantile ties are all retained.
        np.testing.assert_array_equal(model["core_distances"], np.zeros(8))
        for index, neighbors in enumerate(model["neighbor_indices"]):
            self.assertNotIn(index, neighbors)
        np.testing.assert_array_equal(predict_density_regions(data, model), model["labels"])

    def test_invalid_embedding_or_settings_fail(self) -> None:
        data = circle(np.linspace(0, 0.1, 10))
        for kwargs in ({"n_neighbors": 10}, {"n_neighbors": 0}, {"density_quantile": 1.1}, {"min_cluster_size": 0}):
            with self.assertRaises(ValueError):
                fit_density_regions(data, **({"n_neighbors": 3} | kwargs))
        with self.assertRaisesRegex(ValueError, "unit-length"):
            fit_density_regions(data * 2, 3)


if __name__ == "__main__":
    unittest.main()
