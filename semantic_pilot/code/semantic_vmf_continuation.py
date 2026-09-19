"""Checkpointable continuation of the pilot's existing capped vMF EM fits.

Each call advances at most one chunk. A chunk boundary is a resumable state,
never a convergence claim or a total-iteration limit. The caller may continue
chunks and cluster jobs until the original average-likelihood tolerance holds.
The original numerical implementation and the starting parameters are retained.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from semantic_region_directional import (
    _expectation, _maximization, _unit_rows, mean_resultant_length,
)


def _integer(value, name: str, minimum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _nonnegative(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and nonnegative")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _likelihood_matches(observed: float, recorded: float, n: int) -> bool:
    # Permit float64 summation/BLAS differences across hosts, while remaining
    # much tighter than the unchanged 1e-5 average-likelihood stopping rule.
    return bool(abs(observed - recorded) <= n * 1e-8 + abs(recorded) * 1e-12)


def continue_vmf(
    X: np.ndarray,
    model: dict,
    *,
    tol: float = 1e-5,
    kappa_max: Optional[float] = None,
    chunk_iterations: int = 200,
    checkpoint_callback: Optional[Callable[[dict], None]] = None,
    checkpoint_every: int = 10,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> dict:
    """Resume the exact saved mixture without reinitializing its components.

    ``model`` combines the arrays and scalar fields returned by ``fit_vmf``.
    ``iterations`` is cumulative, and history contains the initial likelihood
    followed by every completed EM iteration. ``kappa_max`` defaults to the saved
    cap; changing a saved cap or continuation tolerance is rejected.

    A detached full model is passed to ``checkpoint_callback`` every requested
    checkpoint interval and on exit. The zero-argument ``stop_requested`` is
    checked before each iteration, allowing an external wall-clock deadline to
    stop at a consistent state. An exception from the callback propagates so a
    failed checkpoint cannot be silently mistaken for a durable one.

    Returned fields add ``iterations_added``, ``tolerance``, and ``stop_reason``:
    ``converged``, ``chunk_boundary``, or ``stop_requested``. Intermediate callback
    states use ``checkpoint``. None of the latter three means convergence.
    """
    X = _unit_rows(X)
    n, dimension = X.shape
    tol = _nonnegative(tol, "tol")
    chunk_iterations = _integer(chunk_iterations, "chunk_iterations", 1)
    checkpoint_every = _integer(checkpoint_every, "checkpoint_every", 1)
    if not isinstance(model, dict):
        raise ValueError("model must be a dictionary of saved arrays and scalars")
    required = ("centers", "weights", "kappas", "iterations", "converged",
                "log_likelihood", "log_likelihood_history", "kappa_max")
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"saved vMF model lacks fields: {', '.join(missing)}")
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise ValueError("checkpoint_callback must be callable")
    if stop_requested is not None and not callable(stop_requested):
        raise ValueError("stop_requested must be callable")

    saved_cap = _nonnegative(model["kappa_max"], "saved kappa_max")
    kappa_max = saved_cap if kappa_max is None else _nonnegative(kappa_max, "kappa_max")
    if kappa_max != saved_cap:
        raise ValueError("continuation cannot change the saved concentration cap")
    if "tolerance" in model and _nonnegative(model["tolerance"], "saved tolerance") != tol:
        raise ValueError("continuation cannot change the saved convergence tolerance")
    original_iterations = _integer(model["iterations"], "saved iterations", 0)
    if not isinstance(model["converged"], (bool, np.bool_)):
        raise ValueError("saved converged must be a boolean")

    # Validate unit norms without renormalizing the saved centers: continuation
    # must preserve precisely the parameters produced by the preceding M step.
    centers = np.array(model["centers"], dtype=np.float64, copy=True)
    _unit_rows(centers, "saved centers")
    k = len(centers)
    if centers.shape[1] != dimension or k > n:
        raise ValueError("saved centers do not match the training matrix")
    weights = np.array(model["weights"], dtype=np.float64, copy=True)
    kappas = np.array(model["kappas"], dtype=np.float64, copy=True)
    if weights.shape != (k,) or kappas.shape != (k,):
        raise ValueError("saved weights and kappas must have one entry per component")
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError("saved weights must be finite, nonnegative, and sum to one")
    if (not np.isfinite(kappas).all() or np.any(kappas < 0)
            or np.any(kappas > kappa_max)):
        raise ValueError("saved kappas violate the fixed concentration cap")

    history = np.array(model["log_likelihood_history"], dtype=np.float64, copy=True)
    if (history.ndim != 1 or len(history) != original_iterations + 1
            or not np.isfinite(history).all()):
        raise ValueError("saved likelihood history must contain iterations + 1 finite values")
    if np.any(np.diff(history) / n < -1e-7):
        raise ValueError("saved likelihood history contains an invalid EM decrease")
    saved_likelihood = float(model["log_likelihood"])
    if (not np.isfinite(saved_likelihood)
            or not _likelihood_matches(saved_likelihood, float(history[-1]), n)):
        raise ValueError("saved log likelihood does not match the saved history")
    log_density, responsibilities = _expectation(X, centers, weights, kappas)
    actual_likelihood = float(log_density.sum())
    if not _likelihood_matches(actual_likelihood, saved_likelihood, n):
        raise ValueError("saved log likelihood does not match the supplied parameters and training data")
    labels = np.argmax(responsibilities, axis=1)
    if "labels" in model:
        saved_labels = np.asarray(model["labels"])
        if (saved_labels.shape != (n,) or saved_labels.dtype.kind not in "iu"
                or not np.array_equal(saved_labels, labels)):
            raise ValueError("saved labels do not match the supplied model and training data")
    converged = bool(model["converged"])
    if converged and (original_iterations == 0
                      or (history[-1] - history[-2]) / n > tol + 1e-10):
        raise ValueError("saved convergence claim does not meet the unchanged tolerance")

    history_values = history.tolist()
    previous = saved_likelihood
    iterations = original_iterations
    added = 0

    def snapshot(reason: str) -> dict:
        state = deepcopy(model)
        current_labels = np.argmax(responsibilities, axis=1)
        state.update({
            "centers": centers.copy(),
            "weights": weights.copy(),
            "kappas": kappas.copy(),
            "labels": current_labels,
            "iterations": iterations,
            "iterations_added": added,
            "converged": converged,
            "log_likelihood": previous,
            "log_likelihood_history": np.asarray(history_values, dtype=np.float64),
            "component_sizes": np.bincount(current_labels, minlength=k),
            "effective_component_sizes": responsibilities.sum(axis=0),
            "kappa_max": kappa_max,
            "tolerance": tol,
            "stop_reason": reason,
        })
        return state

    def finish(reason: str) -> dict:
        state = snapshot(reason)
        if checkpoint_callback is not None:
            checkpoint_callback(deepcopy(state))
        return state

    if converged:
        return finish("converged")
    cap_rbar = mean_resultant_length(dimension, kappa_max)
    while added < chunk_iterations:
        if stop_requested is not None and stop_requested():
            return finish("stop_requested")
        updated_centers, updated_weights, updated_kappas = _maximization(
            X, responsibilities, centers, kappa_max, cap_rbar)
        updated_density, updated_responsibilities = _expectation(
            X, updated_centers, updated_weights, updated_kappas)
        current = float(updated_density.sum())
        improvement = (current - previous) / n
        if improvement < -1e-7:
            raise FloatingPointError("capped vMF EM decreased average log likelihood")
        centers, weights, kappas = updated_centers, updated_weights, updated_kappas
        responsibilities = updated_responsibilities
        previous = current
        history_values.append(current)
        added += 1
        iterations += 1
        if improvement <= tol:
            converged = True
            return finish("converged")
        if added == chunk_iterations:
            return finish("chunk_boundary")
        if checkpoint_callback is not None and added % checkpoint_every == 0:
            checkpoint_callback(snapshot("checkpoint"))
    raise AssertionError("positive chunk must exit at convergence, stop, or chunk boundary")
