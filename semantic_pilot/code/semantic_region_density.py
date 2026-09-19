"""Bounded k-nearest-neighbor level sets for normalized post embeddings.

The kth-neighbor radius is a *relative local concentration proxy*, not a fitted
PDF. Its density interpretation requires locally comparable dimension and
sampling; this module does not estimate intrinsic dimension or correct the
community sampling design. A density_quantile of .8 retains points whose
radii are in the lowest 20%, including boundary ties. Lower radii indicate
greater local concentration at the selected neighborhood scale.

The training graph is the symmetrized kNN graph of ALL training points, held
fixed while vertices are filtered by radius. Connected components of this
induced graph define candidate regions; components smaller than the declared
minimum size are noise. This gives nested level sets, although component labels
can change or disappear across thresholds. It is an exploratory graph
construction, not a claim to have recovered the population density cluster tree.

Heldout assignment is a conservative extension: a post must be within the
training kth-neighbor radius of its nearest retained, accepted training point.
This neither refits the graph nor discovers new heldout components. Training
assignments are model['labels']; predicting the training rows is NOT equivalent
to retaining the training level set.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


def _unit_vectors(X: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("X must be a two-dimensional matrix with at least one feature")
    if not allow_empty and len(values) < 1:
        raise ValueError("X must contain at least one row")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("X must contain finite values")
    if len(values) and not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("X must contain normalized, unit-length embeddings")
    return values


def _level_set(
    core_distances: np.ndarray,
    neighbor_indices: np.ndarray,
    density_quantile: float,
    min_cluster_size: int,
) -> dict[str, Any]:
    if not np.isfinite(density_quantile) or not 0 <= density_quantile <= 1:
        raise ValueError("density_quantile must be in [0, 1]")
    if isinstance(min_cluster_size, bool) or int(min_cluster_size) != min_cluster_size or min_cluster_size < 1:
        raise ValueError("min_cluster_size must be a positive integer")
    n_rows, k = neighbor_indices.shape
    threshold = float(np.quantile(core_distances, 1.0 - density_quantile))
    retained = core_distances <= threshold
    active_indices = np.flatnonzero(retained)
    rows = np.repeat(np.arange(n_rows), k)
    columns = neighbor_indices.reshape(-1)
    keep = retained[rows] & retained[columns]
    graph = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.uint8), (rows[keep], columns[keep])),
        shape=(n_rows, n_rows),
    )
    graph = graph.maximum(graph.T)
    subgraph = graph[active_indices][:, active_indices]
    component_count, component_labels = connected_components(subgraph, directed=False)
    raw_sizes = np.bincount(component_labels, minlength=component_count)
    eligible = np.flatnonzero(raw_sizes >= min_cluster_size)
    # scipy numbers components by the first encountered training index; retain
    # that deterministic order rather than introducing a size-based tie breaker.
    labels = np.full(n_rows, -1, dtype=np.int32)
    lookup = np.full(component_count, -1, dtype=np.int32)
    lookup[eligible] = np.arange(len(eligible), dtype=np.int32)
    labels[active_indices] = lookup[component_labels]
    return {
        "labels": labels,
        "retained_mask": retained,
        "threshold_radius": threshold,
        "density_quantile": float(density_quantile),
        "min_cluster_size": int(min_cluster_size),
        "n_retained": int(retained.sum()),
        "n_components_before_min_size": int(component_count),
        "n_regions": int(len(eligible)),
        "component_sizes": [int(raw_sizes[index]) for index in eligible],
        "n_noise": int((labels < 0).sum()),
        "n_assigned": int((labels >= 0).sum()),
        "retained_fraction": float(retained.mean()),
    }


def fit_density_regions(
    X: np.ndarray,
    n_neighbors: int = 30,
    density_quantile: float = 0.5,
    min_cluster_size: int = 30,
) -> dict[str, Any]:
    """Fit radius-filtered regions without allocating a dense n-by-n matrix.

    Euclidean chord distance on unit vectors is used both for graph construction
    and prediction. Exact brute-force neighbor search is quadratic in time but
    chunked in memory; the pilot should keep the training sample bounded. Native
    numerical threads are limited to one to avoid competing parallel fits.
    """
    values = np.array(_unit_vectors(X), copy=True)
    if isinstance(n_neighbors, bool) or int(n_neighbors) != n_neighbors or not 1 <= n_neighbors < len(values):
        raise ValueError("n_neighbors must be an integer from 1 to n_samples - 1")
    search = NearestNeighbors(n_neighbors=int(n_neighbors), metric="euclidean", algorithm="brute", n_jobs=1)
    with threadpool_limits(limits=1), config_context(working_memory=64):
        search.fit(values)
        # X=None explicitly excludes each point itself, including duplicate rows.
        distances, neighbors = search.kneighbors(return_distance=True)
    radii = distances[:, -1].astype(np.float64, copy=False)
    neighbors = neighbors.astype(np.int32, copy=False)
    model = _level_set(radii, neighbors, density_quantile, min_cluster_size)
    model.update({
        "train_X": values,
        "core_distances": radii,
        "neighbor_indices": neighbors,
        "n_neighbors": int(n_neighbors),
        "metric": "euclidean_chord_on_unit_vectors",
        "density_proxy": "smaller_training_kth_neighbor_radius",
        "extension_rule": "nearest_accepted_core_within_that_core_training_kth_radius",
        "normalization": "unit_length_required",
    })
    return model


def density_level_set_summaries(
    model: dict[str, Any], quantiles: Iterable[float] = (0.0, 0.5, 0.8)
) -> list[dict[str, Any]]:
    """Inspect thresholds using the same radii and graph, without refitting kNN.

    Returns only scalar/list summaries so these can be recorded in run JSON.
    Changing thresholds here does not mutate the fitted model or its labels.
    """
    summaries = []
    for quantile in quantiles:
        result = _level_set(model["core_distances"], model["neighbor_indices"], quantile, model["min_cluster_size"])
        summaries.append({key: value for key, value in result.items() if key not in ("labels", "retained_mask")})
    return summaries


def predict_density_regions(X: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Return frozen-region labels, with -1 for rejected heldout posts.

    Only distance to the nearest accepted training core is evaluated. Distances
    to centroids and proportions in the heldout sample do not affect assignment.
    The rejection rule is part of the estimator and coverage must be reported.
    """
    values = _unit_vectors(X, allow_empty=True)
    if values.shape[1] != model["train_X"].shape[1]:
        raise ValueError("Prediction dimension differs from training dimension")
    result = np.full(len(values), -1, dtype=np.int32)
    accepted = np.flatnonzero(model["labels"] >= 0)
    if not len(values) or not len(accepted):
        return result
    search = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute", n_jobs=1)
    with threadpool_limits(limits=1), config_context(working_memory=64):
        search.fit(model["train_X"][accepted])
        distances, nearest = search.kneighbors(values)
    nearest_train = accepted[nearest[:, 0]]
    radii = model["core_distances"][nearest_train]
    # Roundoff tolerance permits exact duplicate vectors whose computed distance
    # is very slightly positive even when the training radius is exactly zero.
    tolerance = 8 * np.finfo(values.dtype).eps
    within = distances[:, 0] <= radii + tolerance
    result[within] = model["labels"][nearest_train[within]]
    return result
