"""Small, bounded directional-clustering pilot on unit post embeddings.

Spherical k-means minimizes mean cosine distance. The vMF mixture uses exact
maximum-likelihood EM updates subject to 0 <= kappa <= kappa_max, with densities
relative to surface-area measure on the unit sphere. Components are not asserted
to be density modes or semantic regions. Both routines fit one seeded start;
the caller is responsible for comparing repeated starts and held-out results.

References:
https://www.jmlr.org/papers/v6/banerjee05a.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, ive, logsumexp


def _matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Some Accelerate/NumPy builds leave floating-point flags set even when a
    # BLAS product of finite unit vectors is finite. Validate the actual result.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = left @ right
    if not np.isfinite(result).all():
        raise FloatingPointError("directional-model matrix product is non-finite")
    return result


def _unit_rows(X: np.ndarray, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError(f"{name} must have shape (n >= 1, dimension >= 2)")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains non-finite entries")
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms == 0) or not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-6):
        raise ValueError(f"{name} must contain nonzero unit vectors")
    # Remove only rounding error in already-normalized input, not arbitrary scale.
    return np.ascontiguousarray(X / norms[:, None])


def _fit_args(X: np.ndarray, k: int, max_iter: int, tol: float) -> None:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X):
        raise ValueError("k must be an integer between 1 and the number of rows")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")


def _seed_centers(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k-means++ sampling: squared Euclidean distance is twice cosine distance."""
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(X), dtype=bool)
    indexes = [int(rng.integers(len(X)))]
    selected[indexes[0]] = True
    nearest = np.maximum(0.0, 1.0 - _matmul(X, X[indexes[0]]))
    for _ in range(1, k):
        nearest[selected] = 0.0
        total = nearest.sum()
        if total <= np.finfo(float).eps:
            index = int(rng.choice(np.flatnonzero(~selected)))
        else:
            index = int(rng.choice(len(X), p=nearest / total))
        indexes.append(index)
        selected[index] = True
        nearest = np.minimum(nearest, np.maximum(0.0, 1.0 - _matmul(X, X[index])))
    return X[indexes].copy()


def predict_spherical(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    X = _unit_rows(X)
    centers = _unit_rows(centers, "centers")
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers have different dimensions")
    return np.argmax(_matmul(X, centers.T), axis=1)


def fit_spherical_kmeans(
    X: np.ndarray, k: int, seed: int, max_iter: int = 100, tol: float = 1e-5
) -> dict:
    """Return one spherical Lloyd fit; objective is mean cosine distance.

    tol is an absolute improvement in mean cosine distance. Empty or exactly
    cancelling groups retain their previous centers; their occupancy is exposed.
    """
    X = _unit_rows(X)
    _fit_args(X, k, max_iter, tol)
    centers = _seed_centers(X, k, seed)
    similarities = _matmul(X, centers.T)
    objective = float(np.mean(1.0 - np.max(similarities, axis=1)))
    history = [objective]
    converged = False
    for iteration in range(1, max_iter + 1):
        labels = np.argmax(similarities, axis=1)
        for j in range(k):
            total = X[labels == j].sum(axis=0)
            norm = np.linalg.norm(total)
            if norm > 0:
                centers[j] = total / norm
        similarities = _matmul(X, centers.T)
        current = float(np.mean(1.0 - np.max(similarities, axis=1)))
        history.append(current)
        if objective - current <= tol:
            converged = True
            objective = current
            break
        objective = current
    labels = np.argmax(similarities, axis=1)
    return {
        "centers": centers,
        "labels": labels,
        "iterations": iteration,
        "converged": converged,
        "objective": objective,
        "objective_history": np.asarray(history),
        "component_sizes": np.bincount(labels, minlength=k),
    }


def _log_hypergeom_series(b: float, kappa: float) -> float:
    """log 0F1(; b; kappa**2/4), using its positive-term convergent series.

    Used only if scaled Bessel functions underflow at high dimension/low kappa.
    Log arithmetic prevents overflow and stops only after the series peak.
    """
    if kappa == 0:
        return 0.0
    log_z = 2.0 * np.log(kappa) - np.log(4.0)
    log_term = 0.0
    log_total = 0.0
    for m in range(1, 100000):
        log_ratio = log_z - np.log(float(m)) - np.log(b + m - 1)
        log_term += log_ratio
        log_total = float(np.logaddexp(log_total, log_term))
        if log_ratio < 0 and log_term < log_total - 36:
            return log_total
    raise FloatingPointError("vMF normalization series did not converge")


def log_vmf_normalizer(dimension: int, kappa: float) -> float:
    """Log C_d(kappa), including the uniform kappa=0 case, in float64."""
    if not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError("dimension must be an integer >= 2")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    half_d = dimension / 2.0
    uniform = float(gammaln(half_d) - np.log(2.0) - half_d * np.log(np.pi))
    if kappa == 0:
        return uniform
    scaled_bessel = ive(half_d - 1.0, kappa)
    if np.isfinite(scaled_bessel) and scaled_bessel > 0:
        return float((half_d - 1.0) * np.log(kappa)
                     - half_d * np.log(2.0 * np.pi)
                     - np.log(scaled_bessel) - kappa)
    return uniform - _log_hypergeom_series(half_d, kappa)


def mean_resultant_length(dimension: int, kappa: float) -> float:
    """A_d(kappa) = I_(d/2)(kappa) / I_(d/2-1)(kappa)."""
    if not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError("dimension must be an integer >= 2")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if kappa == 0:
        return 0.0
    half_d = dimension / 2.0
    denominator = ive(half_d - 1.0, kappa)
    numerator = ive(half_d, kappa)
    if denominator > 0 and numerator > 0 and np.isfinite(denominator + numerator):
        return float(numerator / denominator)
    return float(kappa / dimension * np.exp(
        _log_hypergeom_series(half_d + 1.0, kappa)
        - _log_hypergeom_series(half_d, kappa)))


def _estimate_kappa(rbar: float, dimension: int, cap: float, cap_rbar: float) -> float:
    if rbar <= 0 or cap == 0:
        return 0.0
    if rbar >= cap_rbar:
        return float(cap)
    return float(brentq(lambda kappa: mean_resultant_length(dimension, kappa) - rbar,
                        0.0, cap, xtol=1e-9, rtol=1e-12))


def _expectation(X: np.ndarray, centers: np.ndarray, weights: np.ndarray,
                 kappas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_normalizers = np.asarray([log_vmf_normalizer(X.shape[1], k) for k in kappas])
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    log_joint = _matmul(X, centers.T) * kappas + log_normalizers + log_weights
    log_density = logsumexp(log_joint, axis=1)
    responsibilities = np.exp(log_joint - log_density[:, None])
    if not np.isfinite(log_density).all() or not np.isfinite(responsibilities).all():
        raise FloatingPointError("non-finite vMF expectation")
    return log_density, responsibilities


def _maximization(X: np.ndarray, responsibilities: np.ndarray, centers: np.ndarray,
                  kappa_max: float, cap_rbar: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = responsibilities.sum(axis=0)
    resultants = _matmul(responsibilities.T, X)
    lengths = np.linalg.norm(resultants, axis=1)
    updated_centers = centers.copy()
    positive = lengths > 0
    updated_centers[positive] = resultants[positive] / lengths[positive, None]
    weights = counts / counts.sum()
    kappas = np.zeros(len(weights))
    for j in np.flatnonzero(counts > 0):
        rbar = float(np.clip(lengths[j] / counts[j], 0.0, 1.0))
        kappas[j] = _estimate_kappa(rbar, X.shape[1], kappa_max, cap_rbar)
    return updated_centers, weights, kappas


def fit_vmf(
    X: np.ndarray, k: int, seed: int, kappa_max: float = 1000.0,
    max_iter: int = 100, tol: float = 1e-5
) -> dict:
    """Fit one vMF mixture by EM with concentration bounded by kappa_max.

    tol is an absolute increase in average log likelihood, not relative to the
    dimension-dependent normalization constant. log_likelihood is the total.
    Zero-weight components remain inactive instead of being silently restarted.
    """
    X = _unit_rows(X)
    _fit_args(X, k, max_iter, tol)
    if not np.isfinite(kappa_max) or kappa_max < 0:
        raise ValueError("kappa_max must be finite and nonnegative")
    initial = fit_spherical_kmeans(X, k, seed, max_iter=max_iter, tol=tol)
    responsibilities = np.zeros((len(X), k))
    responsibilities[np.arange(len(X)), initial["labels"]] = 1.0
    cap_rbar = mean_resultant_length(X.shape[1], kappa_max)
    centers, weights, kappas = _maximization(
        X, responsibilities, initial["centers"], kappa_max, cap_rbar)
    log_density, responsibilities = _expectation(X, centers, weights, kappas)
    previous = float(log_density.sum())
    history = [previous]
    converged = False
    for iteration in range(1, max_iter + 1):
        centers, weights, kappas = _maximization(
            X, responsibilities, centers, kappa_max, cap_rbar)
        log_density, responsibilities = _expectation(X, centers, weights, kappas)
        current = float(log_density.sum())
        history.append(current)
        improvement = (current - previous) / len(X)
        if improvement < -1e-7:
            raise FloatingPointError("capped vMF EM decreased average log likelihood")
        if improvement <= tol:
            converged = True
            break
        previous = current
    labels = np.argmax(responsibilities, axis=1)
    return {
        "centers": centers,
        "weights": weights,
        "kappas": kappas,
        "labels": labels,
        "iterations": iteration,
        "converged": converged,
        "log_likelihood": current,
        "log_likelihood_history": np.asarray(history),
        "component_sizes": np.bincount(labels, minlength=k),
        "effective_component_sizes": responsibilities.sum(axis=0),
        "kappa_max": float(kappa_max),
        "initialization_iterations": initial["iterations"],
    }


def predict_vmf(X: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hard component labels, soft memberships, and surface log density."""
    X = _unit_rows(X)
    centers = _unit_rows(model["centers"], "centers")
    weights = np.asarray(model["weights"], dtype=float)
    kappas = np.asarray(model["kappas"], dtype=float)
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers have different dimensions")
    if weights.shape != (len(centers),) or kappas.shape != weights.shape:
        raise ValueError("weights and kappas must have one entry per component")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be finite, nonnegative, and sum to one")
    if not np.isfinite(kappas).all() or np.any(kappas < 0):
        raise ValueError("kappas must be finite and nonnegative")
    log_density, responsibilities = _expectation(X, centers, weights, kappas)
    return np.argmax(responsibilities, axis=1), responsibilities, log_density
