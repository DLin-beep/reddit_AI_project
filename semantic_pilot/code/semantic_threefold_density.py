"""Threaded, reusable exact-neighbor geometry for the threefold density pilot.

The estimator and held-out rejection rule are those in semantic_region_density.
This module changes resource control and reuses one maximum-k neighbor search
across neighborhood sizes and density thresholds. It does not change the
original, authenticated embedding-stage dependency.

For tied neighbor distances, smaller k uses the prefix of the single maximum-k
search, not a new search. scikit-learn chooses boundary ties; the cache preserves
that choice. Consequently a separate k-only search can choose different tied
neighbors. Quantile ties retain every point at the cutoff, as before.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import sklearn
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from semantic_region_density import _level_set, _unit_vectors


TIE_POLICY = "prefix_of_one_maximum_k_search_sklearn_selects_boundary_ties"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _geometry_identity(values: np.ndarray, cache_key: str, max_neighbors: int) -> dict:
    # values is contiguous: memoryview hashes the actual matrix without another
    # n-by-d bytes allocation. Row order is part of the binding.
    return {
        "schema_version": 1,
        "caller_cache_key": cache_key,
        "training_sha256": hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "max_neighbors": max_neighbors,
        "metric": "euclidean_chord_on_unit_vectors",
        "algorithm": "exact_brute_force_self_excluded",
        "tie_policy": TIE_POLICY,
        "sklearn_version": sklearn.__version__,
        "code_sha256": {
            "semantic_threefold_density.py": _sha256(Path(__file__)),
            "semantic_region_density.py": _sha256(Path(__file__).with_name("semantic_region_density.py")),
        },
    }


def _compute_neighbors(values: np.ndarray, maximum: int, threads: int) -> tuple[np.ndarray, np.ndarray]:
    # n_jobs=1 avoids a second joblib fan-out. Native BLAS/OpenMP kernels get the
    # declared thread budget through threadpoolctl instead of being forced to 1.
    search = NearestNeighbors(n_neighbors=maximum, metric="euclidean", algorithm="brute", n_jobs=1)
    with threadpool_limits(limits=threads), config_context(working_memory=64):
        search.fit(values)
        distances, neighbors = search.kneighbors(return_distance=True)
    return distances.astype(np.float64, copy=False), neighbors.astype(np.int32, copy=False)


def _validate_neighbors(distances: np.ndarray, neighbors: np.ndarray, rows: int, maximum: int) -> None:
    expected = (rows, maximum)
    if distances.shape != expected or neighbors.shape != expected:
        raise ValueError("Neighbor cache shape mismatch")
    if not np.issubdtype(distances.dtype, np.floating) or not np.isfinite(distances).all():
        raise ValueError("Neighbor cache has invalid distances")
    if np.any(distances < 0) or np.any(np.diff(distances, axis=1) < 0):
        raise ValueError("Neighbor cache distances are negative or unsorted")
    if not np.issubdtype(neighbors.dtype, np.integer) or np.any(neighbors < 0) or np.any(neighbors >= rows):
        raise ValueError("Neighbor cache has invalid indices")
    if np.any(neighbors == np.arange(rows)[:, None]):
        raise ValueError("Neighbor cache contains self-neighbors")
    if maximum > 1 and np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise ValueError("Neighbor cache contains duplicate indices within a row")


def _cached_neighbors(values: np.ndarray, maximum: int, threads: int,
                      directory: Path, cache_key: str) -> tuple[np.ndarray, np.ndarray, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    identity = _geometry_identity(values, cache_key, maximum)
    digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
    target = directory / f"neighbors_{digest}"
    info = {"neighbor_cache_identity_sha256": digest, "neighbor_cache_path": str(target)}
    # A directory is committed atomically only after both the data and its
    # checksum manifest exist. Competing array tasks wait on the same lock.
    with (directory / f"neighbors_{digest}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            if not target.is_dir() or target.is_symlink():
                raise ValueError("Neighbor cache path is not a regular directory")
            metadata_path, payload = target / "manifest.json", target / "neighbors.npz"
            if metadata_path.is_symlink() or payload.is_symlink():
                raise ValueError("Neighbor cache files must not be symbolic links")
            try:
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("identity") != identity:
                    raise ValueError("Neighbor cache identity mismatch")
                if _sha256(payload) != metadata.get("payload_sha256"):
                    raise ValueError("Neighbor cache checksum mismatch")
                with np.load(payload, allow_pickle=False) as arrays:
                    distances, neighbors = arrays["distances"], arrays["neighbors"]
                _validate_neighbors(distances, neighbors, len(values), maximum)
            except (OSError, KeyError, TypeError, EOFError) as exc:
                raise ValueError("Neighbor cache is incomplete or unreadable") from exc
            return distances, neighbors, dict(info, neighbor_cache_hit=True)
        distances, neighbors = _compute_neighbors(values, maximum, threads)
        _validate_neighbors(distances, neighbors, len(values), maximum)
        temporary = Path(tempfile.mkdtemp(prefix=f".neighbors_{digest}_", dir=directory))
        try:
            payload = temporary / "neighbors.npz"
            with payload.open("wb") as handle:
                np.savez(handle, distances=distances, neighbors=neighbors)
                handle.flush()
                os.fsync(handle.fileno())
            metadata = {"identity": identity, "payload_sha256": _sha256(payload), "created_with_threads": threads}
            with (temporary / "manifest.json").open("w") as handle:
                handle.write(_canonical(metadata) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return distances, neighbors, dict(info, neighbor_cache_hit=False)


def fit_density_regions(
    X: np.ndarray, n_neighbors: int = 30, density_quantile: float = 0.5,
    min_cluster_size: int = 30, threads: int = 16,
    neighbor_cache: Path | None = None, cache_key: str | None = None,
    max_neighbors: int = 30,
) -> dict[str, Any]:
    """Fit the original radius-filtered kNN regions with a native thread budget.

    neighbor_cache is a directory shared across fits. Its caller cache_key must
    bind the ordered training IDs and embedding identity; the module also binds
    every matrix byte, shape, dtype, implementation, and maximum neighborhood.
    A mismatched identity creates a different cache; corrupt caches fail closed.
    max_neighbors must be at least n_neighbors and is clipped to n_samples-1.
    Without a cache, only the requested n_neighbors are computed.
    """
    threads = _positive_integer(threads, "threads")
    k = _positive_integer(n_neighbors, "n_neighbors")
    maximum = _positive_integer(max_neighbors, "max_neighbors")
    minimum = _positive_integer(min_cluster_size, "min_cluster_size")
    if not np.isfinite(density_quantile) or not 0 <= density_quantile <= 1:
        raise ValueError("density_quantile must be in [0, 1]")
    values = np.array(_unit_vectors(X), copy=True, order="C")
    if k >= len(values):
        raise ValueError("n_neighbors must be smaller than n_samples")
    info = {"neighbor_cache_hit": False}
    if neighbor_cache is None:
        maximum = k
        distances, neighbors = _compute_neighbors(values, maximum, threads)
    else:
        if maximum < k:
            raise ValueError("max_neighbors must be at least n_neighbors")
        if not isinstance(cache_key, str) or not cache_key.strip():
            raise ValueError("A nonempty cache_key is required with neighbor_cache")
        maximum = min(maximum, len(values) - 1)
        distances, neighbors, info = _cached_neighbors(values, maximum, threads, Path(neighbor_cache), cache_key)
    radii = distances[:, k - 1].copy()
    selected_neighbors = neighbors[:, :k].copy()
    model = _level_set(radii, selected_neighbors, density_quantile, minimum)
    model.update({
        "train_X": values, "core_distances": radii, "neighbor_indices": selected_neighbors,
        "n_neighbors": k, "metric": "euclidean_chord_on_unit_vectors",
        "density_proxy": "smaller_training_kth_neighbor_radius",
        "extension_rule": "nearest_accepted_core_within_that_core_training_kth_radius",
        "normalization": "unit_length_required", "threads": threads,
        "neighbors_cached": maximum, "neighbor_tie_policy": TIE_POLICY, **info,
    })
    return model


def predict_density_regions(X: np.ndarray, model: dict[str, Any], threads: int = 16) -> np.ndarray:
    """Apply the original frozen nearest-accepted-core rejection rule."""
    threads = _positive_integer(threads, "threads")
    values = _unit_vectors(X, allow_empty=True)
    if values.shape[1] != model["train_X"].shape[1]:
        raise ValueError("Prediction dimension differs from training dimension")
    result = np.full(len(values), -1, dtype=np.int32)
    accepted = np.flatnonzero(model["labels"] >= 0)
    if not len(values) or not len(accepted):
        return result
    search = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute", n_jobs=1)
    with threadpool_limits(limits=threads), config_context(working_memory=64):
        search.fit(model["train_X"][accepted])
        distances, nearest = search.kneighbors(values)
    nearest_train = accepted[nearest[:, 0]]
    tolerance = 8 * np.finfo(values.dtype).eps
    within = distances[:, 0] <= model["core_distances"][nearest_train] + tolerance
    result[within] = model["labels"][nearest_train[within]]
    return result
