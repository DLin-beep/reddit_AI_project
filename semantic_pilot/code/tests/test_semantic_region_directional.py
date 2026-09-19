"""Numerical and synthetic checks for the bounded semantic-region pilot."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import vonmises_fisher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_region_directional import (
    fit_spherical_kmeans, fit_vmf, log_vmf_normalizer,
    mean_resultant_length, predict_spherical, predict_vmf,
)


class DirectionalPilotTests(unittest.TestCase):
    def test_surface_density_integrates_to_one_on_sphere(self):
        # On S^2, surface integration reduces to 2*pi*integral[-1,1] f(z) dz.
        for kappa in (0.0, 1e-5, 1.0, 30.0, 1000.0):
            log_c = log_vmf_normalizer(3, kappa)
            integral = 2 * np.pi * quad(lambda z: np.exp(log_c + kappa * z), -1, 1)[0]
            self.assertAlmostEqual(integral, 1.0, places=9)

    def test_high_dimension_normalizer_and_bessel_ratio(self):
        d = 768
        log_uniform = gammaln(d / 2) - np.log(2) - d / 2 * np.log(np.pi)
        self.assertEqual(log_vmf_normalizer(d, 0), log_uniform)
        # This concentration makes scipy.special.ive underflow in dimension 768.
        self.assertAlmostEqual(mean_resultant_length(d, 1e-4), 1e-4 / d, places=16)
        for kappa in (0.1, 10.0, 100.0, 500.0, 1000.0):
            step = 1e-3
            derivative = (log_vmf_normalizer(d, kappa + step)
                          - log_vmf_normalizer(d, kappa - step)) / (2 * step)
            self.assertAlmostEqual(-derivative, mean_resultant_length(d, kappa), places=7)
        mu = np.zeros(d)
        mu[0] = 1
        # Independent SciPy evaluation is stable at this concentration.
        self.assertAlmostEqual(log_vmf_normalizer(d, 1000) + 1000,
                               float(vonmises_fisher.logpdf(mu, mu, 1000)), places=9)

    def test_one_component_m_step_and_concentration_cap(self):
        X = np.asarray([[0.5, np.sqrt(0.75), 0], [0.5, -np.sqrt(0.75), 0]])
        model = fit_vmf(X, 1, seed=4)
        self.assertAlmostEqual(mean_resultant_length(3, model["kappas"][0]), 0.5, places=10)
        np.testing.assert_allclose(model["centers"], [[1, 0, 0]], atol=1e-12)
        concentrated = np.zeros((12, 768))
        concentrated[:, 0] = 1
        model = fit_vmf(concentrated, 2, seed=4, kappa_max=73)
        self.assertLessEqual(model["kappas"].max(), 73)
        self.assertEqual(model["kappas"][model["weights"] > 0][0], 73)
        self.assertTrue(np.isfinite(model["log_likelihood"]))
        labels, probabilities, density = predict_vmf(concentrated, model)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertTrue(np.isfinite(density).all())

    def test_em_reproducibility_monotonicity_and_separated_groups(self):
        rng = np.random.default_rng(21)
        first = vonmises_fisher.rvs([1., 0, 0], 30, size=100, random_state=rng)
        second = vonmises_fisher.rvs([0., 1, 0], 30, size=100, random_state=rng)
        X = np.vstack([first, second])
        spherical = fit_spherical_kmeans(X, 2, seed=7)
        np.testing.assert_array_equal(spherical["labels"], predict_spherical(X, spherical["centers"]))
        self.assertTrue(np.all(np.diff(spherical["objective_history"]) <= 1e-12))
        first_model = fit_vmf(X, 2, seed=7)
        repeated = fit_vmf(X, 2, seed=7)
        np.testing.assert_allclose(first_model["centers"], repeated["centers"])
        np.testing.assert_allclose(first_model["kappas"], repeated["kappas"])
        self.assertTrue(np.all(np.diff(first_model["log_likelihood_history"]) >= -1e-8))
        labels, probabilities, density = predict_vmf(X, first_model)
        np.testing.assert_array_equal(first_model["labels"], labels)
        self.assertAlmostEqual(first_model["log_likelihood"], density.sum(), places=9)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        accuracy = max(np.mean(labels == np.repeat([0, 1], 100)),
                       np.mean(labels == np.repeat([1, 0], 100)))
        self.assertGreater(accuracy, 0.98)

    def test_uniform_zero_cap_and_bad_input(self):
        X = np.eye(3)
        model = fit_vmf(X, 2, seed=0, kappa_max=0)
        _, probabilities, density = predict_vmf(X, model)
        np.testing.assert_allclose(density, -np.log(4 * np.pi))
        np.testing.assert_allclose(probabilities, np.tile(model["weights"], (3, 1)))
        for invalid in (np.zeros((3, 3)), np.ones((3, 3)), np.asarray([[np.nan, 1]]), np.ones(3)):
            with self.assertRaises(ValueError):
                fit_spherical_kmeans(invalid, 1, seed=0)
            with self.assertRaises(ValueError):
                fit_vmf(invalid, 1, seed=0)
        with self.assertRaises(ValueError):
            fit_vmf(X, 1, seed=0, kappa_max=np.inf)
        with self.assertRaises(ValueError):
            fit_vmf(X, 4, seed=0)


if __name__ == "__main__":
    unittest.main()
