"""Continuation preserves EM state, stopping rules, and durable checkpoints."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_region_directional import fit_vmf, _expectation, _maximization, _unit_rows, mean_resultant_length
from semantic_vmf_continuation import continue_vmf


class VmfContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(47)
        X = rng.normal(size=(180, 7))
        X[:90, 0] += 1.8
        X[90:, 1] += 1.8
        cls.X = X / np.linalg.norm(X, axis=1)[:, None]
        cls.initial = fit_vmf(cls.X, 3, 19, kappa_max=40, max_iter=1, tol=1e-9)
        assert not cls.initial["converged"]

    def assertModelsEqual(self, left, right):
        for key in ("centers", "weights", "kappas", "log_likelihood_history",
                    "effective_component_sizes"):
            np.testing.assert_allclose(left[key], right[key], rtol=1e-12, atol=1e-12)
        for key in ("labels", "component_sizes"):
            np.testing.assert_array_equal(left[key], right[key])
        self.assertEqual(left["iterations"], right["iterations"])
        self.assertEqual(left["converged"], right["converged"])

    def test_chunked_and_uninterrupted_reach_same_converged_state(self):
        uninterrupted = continue_vmf(self.X, self.initial, chunk_iterations=5000)
        self.assertTrue(uninterrupted["converged"])
        chunked = deepcopy(self.initial)
        calls = 0
        while not chunked["converged"]:
            chunked = continue_vmf(self.X, chunked, chunk_iterations=3)
            calls += 1
            self.assertLess(calls, 2000, "synthetic fixture must converge")
        self.assertGreater(calls, 1)
        self.assertModelsEqual(uninterrupted, chunked)
        self.assertEqual(chunked["stop_reason"], "converged")

    def test_exact_reference_em_steps_and_no_false_chunk_convergence(self):
        X = _unit_rows(self.X)
        centers = self.initial["centers"].copy()
        weights = self.initial["weights"].copy()
        kappas = self.initial["kappas"].copy()
        _, responsibilities = _expectation(X, centers, weights, kappas)
        history = self.initial["log_likelihood_history"].tolist()
        for _ in range(7):
            centers, weights, kappas = _maximization(
                X, responsibilities, centers, 40, mean_resultant_length(X.shape[1], 40))
            density, responsibilities = _expectation(X, centers, weights, kappas)
            history.append(float(density.sum()))
        actual = continue_vmf(self.X, self.initial, chunk_iterations=7, tol=0)
        self.assertFalse(actual["converged"])
        self.assertEqual(actual["iterations_added"], 7)
        self.assertEqual(actual["iterations"], self.initial["iterations"] + 7)
        self.assertEqual(actual["stop_reason"], "chunk_boundary")
        np.testing.assert_allclose(actual["log_likelihood_history"], history, atol=1e-12)
        np.testing.assert_allclose(actual["centers"], centers, atol=1e-12)

    def test_checkpoint_can_restore_after_interruption(self):
        checkpoints = []
        class SimulatedPreemption(Exception):
            pass

        def checkpoint(state):
            checkpoints.append(deepcopy(state))
            raise SimulatedPreemption()

        with self.assertRaises(SimulatedPreemption):
            continue_vmf(self.X, self.initial, checkpoint_every=4, checkpoint_callback=checkpoint)
        self.assertEqual(checkpoints[-1]["iterations_added"], 4)
        resumed = continue_vmf(self.X, checkpoints[-1], chunk_iterations=5000)
        uninterrupted = continue_vmf(self.X, self.initial, chunk_iterations=5000)
        self.assertModelsEqual(resumed, uninterrupted)

    def test_wall_stop_before_first_step_and_after_checkpoint(self):
        saved = []
        stopped = continue_vmf(self.X, self.initial, stop_requested=lambda: True,
                               checkpoint_callback=saved.append)
        self.assertEqual(stopped["iterations_added"], 0)
        self.assertEqual(stopped["stop_reason"], "stop_requested")
        self.assertFalse(stopped["converged"])
        self.assertModelsEqual(stopped, self.initial)
        self.assertEqual(len(saved), 1)
        saved.clear()
        stopped = continue_vmf(self.X, self.initial, checkpoint_every=2,
                               checkpoint_callback=saved.append,
                               stop_requested=lambda: len(saved) > 0)
        self.assertEqual(stopped["iterations_added"], 2)
        self.assertEqual(stopped["stop_reason"], "stop_requested")
        self.assertEqual(saved[-1]["stop_reason"], "stop_requested")

    def test_converged_state_is_verified_then_preserved(self):
        model = continue_vmf(self.X, self.initial, chunk_iterations=5000)
        with patch("semantic_vmf_continuation._maximization", side_effect=AssertionError("no M step")):
            no_op = continue_vmf(self.X, model)
        self.assertModelsEqual(model, no_op)
        self.assertEqual(no_op["iterations_added"], 0)
        corrupt = deepcopy(model)
        corrupt["log_likelihood"] += 1
        with self.assertRaisesRegex(ValueError, "likelihood"):
            continue_vmf(self.X, corrupt)

    def test_reject_corrupt_likelihood_data_history_labels_and_parameters(self):
        corruptions = [
            ("log_likelihood", self.initial["log_likelihood"] + 1),
            ("log_likelihood_history", self.initial["log_likelihood_history"] + 1),
            ("iterations", self.initial["iterations"] + 1),
            ("converged", True),
            ("weights", np.full(3, .4)),
            ("kappas", np.full(3, 41.)),
            ("labels", np.full(len(self.X), -1)),
        ]
        for name, value in corruptions:
            with self.subTest(name=name):
                model = deepcopy(self.initial)
                model[name] = value
                with self.assertRaises(ValueError):
                    continue_vmf(self.X, model)
        with self.assertRaisesRegex(ValueError, "training data"):
            continue_vmf(self.X[::-1], self.initial)

    def test_does_not_change_cap_tolerance_or_mutate_original(self):
        original = deepcopy(self.initial)
        with self.assertRaisesRegex(ValueError, "concentration cap"):
            continue_vmf(self.X, self.initial, kappa_max=4000)
        continued = continue_vmf(self.X, self.initial, chunk_iterations=2)
        with self.assertRaisesRegex(ValueError, "convergence tolerance"):
            continue_vmf(self.X, continued, tol=1e-4)
        self.assertModelsEqual(self.initial, original)
        self.assertNotIn("stop_reason", self.initial)

    def test_callback_mutations_do_not_change_trajectory(self):
        def malicious_callback(state):
            state["centers"][:] = 0
            state["log_likelihood_history"][:] = 0
            state["weights"][:] = 0
            state["iterations"] = 0

        actual = continue_vmf(self.X, self.initial, chunk_iterations=9,
                              checkpoint_every=2, checkpoint_callback=malicious_callback)
        expected = continue_vmf(self.X, self.initial, chunk_iterations=9)
        self.assertModelsEqual(actual, expected)

    def test_invalid_limits_and_saved_fields_are_rejected(self):
        for kwargs in ({"chunk_iterations": 0}, {"chunk_iterations": True},
                       {"checkpoint_every": 0}, {"tol": np.nan},
                       {"tol": -1}, {"stop_requested": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                continue_vmf(self.X, self.initial, **kwargs)
        model = deepcopy(self.initial)
        del model["kappa_max"]
        with self.assertRaisesRegex(ValueError, "lacks fields"):
            continue_vmf(self.X, model)

    def test_decreasing_update_is_not_checkpointed(self):
        saved = []
        from semantic_vmf_continuation import _expectation as real_expectation
        calls = 0

        def decreasing(*args):
            nonlocal calls
            calls += 1
            density, responsibilities = real_expectation(*args)
            return (density - 1000 if calls > 1 else density), responsibilities

        with patch("semantic_vmf_continuation._expectation", side_effect=decreasing):
            with self.assertRaises(FloatingPointError):
                continue_vmf(self.X, self.initial, checkpoint_callback=saved.append)
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
