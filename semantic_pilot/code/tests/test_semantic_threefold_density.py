"""Checks for threaded density equivalence and authenticated geometry reuse."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import semantic_threefold_density as density
from semantic_region_density import fit_density_regions as old_fit, predict_density_regions as old_predict


def data(seed=7, rows=40):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 6))
    values[:rows // 2, 0] += 5
    values[rows // 2:, 0] -= 5
    return values / np.linalg.norm(values, axis=1)[:, None]


class ThreefoldDensityTests(unittest.TestCase):
    def test_no_cache_matches_original_geometry_and_predictions(self):
        X, heldout = data(), data(22)
        original = old_fit(X, 5, 0.4, 3)
        actual = density.fit_density_regions(X, 5, 0.4, 3, threads=2)
        np.testing.assert_allclose(actual['core_distances'], original['core_distances'])
        np.testing.assert_array_equal(actual['neighbor_indices'], original['neighbor_indices'])
        np.testing.assert_array_equal(actual['labels'], original['labels'])
        np.testing.assert_array_equal(density.predict_density_regions(heldout, actual, threads=2),
                                      old_predict(heldout, original))

    def test_maximum_geometry_is_reused_for_k_and_quantile(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(density, '_compute_neighbors', wraps=density._compute_neighbors) as compute:
                first = density.fit_density_regions(data(), 3, 0.3, 2, threads=1,
                    neighbor_cache=Path(folder), cache_key='ordered_ids_encoder', max_neighbors=7)
                second = density.fit_density_regions(data(), 7, 0.3, 2, threads=1,
                    neighbor_cache=Path(folder), cache_key='ordered_ids_encoder', max_neighbors=7)
                third = density.fit_density_regions(data(), 3, 0.8, 2, threads=1,
                    neighbor_cache=Path(folder), cache_key='ordered_ids_encoder', max_neighbors=7)
            self.assertEqual(compute.call_count, 1)
            self.assertFalse(first['neighbor_cache_hit'])
            self.assertTrue(second['neighbor_cache_hit'])
            self.assertTrue(third['neighbor_cache_hit'])
            np.testing.assert_array_equal(first['neighbor_indices'], second['neighbor_indices'][:, :3])
            np.testing.assert_array_equal(first['core_distances'], third['core_distances'])
            self.assertTrue(np.all(~third['retained_mask'] | first['retained_mask']))

    def test_matrix_order_content_and_caller_identity_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            def fit(X, key):
                return density.fit_density_regions(X, 3, 0.5, 2, threads=1,
                    neighbor_cache=Path(folder), cache_key=key, max_neighbors=5)
            first = fit(data(), 'a')
            other_key = fit(data(), 'b')
            reordered = fit(data()[::-1], 'a')
            changed = fit(data(33), 'a')
            self.assertEqual(len({x['neighbor_cache_identity_sha256']
                                  for x in (first, other_key, reordered, changed)}), 4)
            self.assertFalse(any(x['neighbor_cache_hit'] for x in (first, other_key, reordered, changed)))
            self.assertTrue(fit(data(), 'a')['neighbor_cache_hit'])

    def test_corrupt_cache_fails_before_using_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(n_neighbors=3, min_cluster_size=2, threads=1,
                          neighbor_cache=Path(folder), cache_key='same', max_neighbors=5)
            first = density.fit_density_regions(data(), **kwargs)
            payload = Path(first['neighbor_cache_path']) / 'neighbors.npz'
            with payload.open('r+b') as handle:
                handle.seek(20)
                handle.write(b'corrupted bytes')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                density.fit_density_regions(data(), **kwargs)

    def test_identity_and_shape_validated_even_with_valid_payload_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(n_neighbors=3, min_cluster_size=2, threads=1,
                          neighbor_cache=Path(folder), cache_key='same', max_neighbors=5)
            first = density.fit_density_regions(data(), **kwargs)
            root = Path(first['neighbor_cache_path'])
            metadata_path = root / 'manifest.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['identity']['caller_cache_key'] = 'wrong'
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                density.fit_density_regions(data(), **kwargs)
            metadata['identity']['caller_cache_key'] = 'same'
            payload = root / 'neighbors.npz'
            np.savez(payload, distances=np.zeros((2, 5)), neighbors=np.zeros((2, 5), dtype=int))
            metadata['payload_sha256'] = hashlib.sha256(payload.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                density.fit_density_regions(data(), **kwargs)

    def test_competing_fits_compute_once_and_commit_both_files(self):
        with tempfile.TemporaryDirectory() as folder:
            def fit(_):
                return density.fit_density_regions(data(), 3, 0.5, 2, threads=1,
                    neighbor_cache=Path(folder), cache_key='shared', max_neighbors=5)
            with patch.object(density, '_compute_neighbors', wraps=density._compute_neighbors) as compute:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first, second = list(pool.map(fit, range(2)))
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(sorted([first['neighbor_cache_hit'], second['neighbor_cache_hit']]), [False, True])
            np.testing.assert_array_equal(first['labels'], second['labels'])
            self.assertEqual({p.name for p in Path(first['neighbor_cache_path']).iterdir()},
                             {'neighbors.npz', 'manifest.json'})

    def test_thread_budget_is_honored_in_fit_and_prediction(self):
        with patch.object(density, 'threadpool_limits', wraps=density.threadpool_limits) as limits:
            model = density.fit_density_regions(data(), 3, 0, 2, threads=2)
            density.predict_density_regions(data(22), model, threads=3)
        self.assertEqual([item.kwargs['limits'] for item in limits.call_args_list], [2, 3])
        self.assertEqual(model['threads'], 2)

    def test_duplicates_empty_prediction_and_input_validation(self):
        X = np.array([[1., 0.]] * 4 + [[-1., 0.]] * 4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            model = density.fit_density_regions(X, 2, 0.8, 3, threads=1,
                neighbor_cache=Path(folder), cache_key='ties')
        self.assertEqual(model['n_regions'], 2)
        self.assertEqual(model['neighbors_cached'], 7)
        np.testing.assert_array_equal(model['core_distances'], np.zeros(8))
        self.assertFalse(np.any(model['neighbor_indices'] == np.arange(8)[:, None]))
        self.assertEqual(density.predict_density_regions(np.empty((0, 2)), model).shape, (0,))
        for kwargs in ({'threads': 0}, {'threads': True}, {'n_neighbors': 40},
                       {'density_quantile': 1.1}, {'min_cluster_size': 0}):
            with self.assertRaises(ValueError):
                density.fit_density_regions(data(), **({'n_neighbors': 3} | kwargs))
        with self.assertRaisesRegex(ValueError, 'unit-length'):
            density.fit_density_regions(data() * 2, 3)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'cache_key'):
                density.fit_density_regions(data(), 3, neighbor_cache=Path(folder))
            with self.assertRaisesRegex(ValueError, 'max_neighbors'):
                density.fit_density_regions(data(), 6, neighbor_cache=Path(folder), cache_key='x', max_neighbors=5)


if __name__ == '__main__':
    unittest.main()
