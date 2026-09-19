import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_semantic_threefold_pilot as pilot


class ThreefoldPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'results'
        self.config = {
            'input': {'index': 'index.csv', 'manifest': 'manifest.json', 'roles': 'roles.csv',
                      'human_reviews': 'reviews.csv', 'shards': 'shards', 'dimension': 3},
            'sampling': {'seed': 20260914, 'expected_posts': 24, 'expected_communities': 2,
                         'expected_posts_per_community': 12, 'expected_human_reviews': 6},
            'models': {'families': ['spherical_kmeans', 'vmf', 'density'], 'k': [2], 'seeds': [11],
                       'density_neighbors': [2], 'density_quantiles': [.5], 'density_min_region_size': 2,
                       'max_iter': 20, 'tolerance': 1e-5, 'vmf_kappa_max': 20},
            'resources': {'threads': 1, 'fit_wall_limit_seconds': 60},
        }
        rows = []
        rng = np.random.default_rng(6)
        vectors = []
        for j in range(24):
            row = dict(annotation_id=f'annotation_{j:03d}', post_id=f'post_{j}', text_sha256=f'{j:064x}',
                       subreddit=f'community_{j // 12}', year_month='2020-01', shard='one.npz', shard_row=j)
            rows.append(row)
            vector = np.array([1 if j % 2 else -1, .2, 0]) + rng.normal(0, .04, 3)
            vectors.append(vector / np.linalg.norm(vector))
        pilot.write_csv(self.root / 'index.csv', rows)
        pilot.write_csv(self.root / 'roles.csv', [dict(annotation_id=row['annotation_id'], post_id=row['post_id'],
                         text_sha256=row['text_sha256'], sample_role='training' if j % 3 == 0 else 'heldout_historical') for j, row in enumerate(rows)])
        pilot.write_csv(self.root / 'reviews.csv', [dict(annotation_id=row['annotation_id'], post_id=row['post_id'],
                         text_sha256=row['text_sha256'], derek_personal_req_score=str(j % 4), arya_personal_req_score=str((j + 1) % 4))
                         for j, row in enumerate(rows[:6])])
        (self.root / 'shards').mkdir()
        np.savez_compressed(self.root / 'shards/one.npz', cache_key=np.asarray('test_cache'),
                            annotation_ids=np.asarray([row['annotation_id'] for row in rows]),
                            text_sha256=np.asarray([row['text_sha256'] for row in rows]), embeddings=np.asarray(vectors))
        pilot.write_json(self.root / 'manifest.json', {'records': 24, 'cache_key': 'test_cache',
                         'encoder': {'embedding_dimension': 3, 'normalize_embeddings': True},
                         'completed_shards': {'one.npz': pilot.sha256(self.root / 'shards/one.npz')}})
        self.config['input_sha256'] = {key: pilot.sha256(self.root / name) for key, name in self.config['input'].items()
                                        if key in ('index', 'manifest', 'roles', 'human_reviews')}

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        return pilot.load_inputs(self.config, self.root)

    def test_balanced_folds_are_deterministic_disjoint_complete_and_ignore_roles(self):
        inputs = self.load()
        folds = inputs['folds']
        self.assertEqual({fold: len(ids) for fold, ids in folds.items()}, {'A': 8, 'B': 8, 'C': 8})
        self.assertEqual(set.union(*(set(ids) for ids in folds.values())), set(inputs['index']))
        for first in pilot.FOLDS:
            for second in pilot.FOLDS:
                if first != second:
                    self.assertFalse(set(folds[first]) & set(folds[second]))
            counts = {community: sum(inputs['index'][aid]['subreddit'] == community for aid in folds[first])
                      for community in ('community_0', 'community_1')}
            self.assertEqual(counts, {'community_0': 4, 'community_1': 4})
        self.assertEqual(folds, pilot.make_folds(dict(reversed(list(inputs['index'].items()))), self.config['sampling']))
        flipped = pilot.read_csv(self.root / 'roles.csv')
        for row in flipped:
            row['sample_role'] = 'heldout_historical' if row['sample_role'] == 'training' else 'training'
        pilot.write_csv(self.root / 'roles.csv', flipped)
        self.config['input_sha256']['roles'] = pilot.sha256(self.root / 'roles.csv')
        self.assertEqual(folds, self.load()['folds'])

    def test_default_grid_has_72_tasks_and_rotated_disjoint_evaluation(self):
        config = json.loads(json.dumps(self.config))
        config['models'].update(k=[10, 20], seeds=[11, 29], density_neighbors=[15, 30], density_quantiles=[.5, .8])
        tasks = pilot.planned_tasks(config)
        self.assertEqual(len(tasks), 72)
        self.assertEqual(len({task['fit_id'] for task in tasks}), 72)
        for task in tasks:
            self.assertFalse(set(task['train_folds']) & set(task['eval_folds']))
            self.assertEqual(set(task['train_folds']) | set(task['eval_folds']), set(pilot.FOLDS))
        self.assertEqual(sum(len(task['eval_folds']) for task in tasks), 108)

    def test_changed_input_and_unbalanced_corpus_rejected(self):
        with (self.root / 'reviews.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.load()
        self.config['input_sha256']['human_reviews'] = pilot.sha256(self.root / 'reviews.csv')
        inputs = self.load()
        del inputs['index']['annotation_000']
        with self.assertRaisesRegex(ValueError, 'count'):
            pilot.make_folds(inputs['index'], self.config['sampling'])

    def test_duplicate_physical_post_ids_fail_even_with_unique_annotation_ids(self):
        rows = pilot.read_csv(self.root / 'index.csv')
        rows[1]['post_id'] = rows[0]['post_id']
        pilot.write_csv(self.root / 'index.csv', rows)
        self.config['input_sha256']['index'] = pilot.sha256(self.root / 'index.csv')
        with self.assertRaisesRegex(ValueError, 'Duplicate post_id'):
            self.load()

    def test_tasks_persist_predictions_review_bindings_and_pickle_free_models(self):
        inputs = self.load()
        summary = pilot.initialize_run(self.config, inputs, self.output)
        # The first three tasks exercise all three estimator families on A.
        for task_id in range(3):
            self.assertEqual(pilot.run_task(self.config, inputs, self.output, task_id, summary), 0)
            task = pilot.planned_tasks(self.config)[task_id]
            task_dir = self.output / 'tasks' / f'task_{task_id:03d}'
            self.assertTrue(pilot.validate_completed_task(task_dir, task, summary))
            with np.load(task_dir / 'model.npz', allow_pickle=False) as model:
                self.assertNotIn('train_X', model.files)
                self.assertEqual(len(model['training_ids']), 8)
                for name in model.files:
                    self.assertFalse(model[name].dtype.hasobject)
            with np.load(task_dir / 'predictions.npz', allow_pickle=False) as arrays:
                training = dict(zip(arrays['training_ids'].tolist(), arrays['training_labels'].tolist()))
                self.assertEqual(set(arrays['evaluation_ids_B']), set(inputs['folds']['B']))
                self.assertEqual(set(arrays['evaluation_ids_C']), set(inputs['folds']['C']))
                self.assertNotIn('evaluation_ids_A', arrays.files)
                self.assertEqual(len(arrays['review_ids']), 6)
                for aid, label, in_fit in zip(arrays['review_ids'], arrays['review_labels'], arrays['review_used_in_fit']):
                    self.assertEqual(bool(in_fit), aid in training)
                    if in_fit:
                        self.assertEqual(label, training[aid])
            review_rows = pilot.read_csv(task_dir / 'review_assignments_restricted.csv')
            self.assertEqual(len(review_rows), 6)
            for row in review_rows:
                source = inputs['reviews'][row['annotation_id']]
                self.assertEqual(row['derek_personal_req_score'], source['derek_personal_req_score'])
                self.assertEqual(row['arya_personal_req_score'], source['arya_personal_req_score'])

    def test_completed_task_reused_without_refit_and_corruption_fails(self):
        inputs = self.load()
        summary = pilot.initialize_run(self.config, inputs, self.output)
        self.assertEqual(pilot.run_task(self.config, inputs, self.output, 0, summary), 0)
        task_dir = self.output / 'tasks/task_000'
        old_hash = pilot.sha256(task_dir / 'status.json')
        with patch.object(pilot, 'execute_fit', side_effect=AssertionError('must not fit again')):
            self.assertEqual(pilot.run_task(self.config, inputs, self.output, 0, summary), 0)
        self.assertEqual(old_hash, pilot.sha256(task_dir / 'status.json'))
        with (task_dir / 'model.npz').open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'modified'):
            pilot.run_task(self.config, inputs, self.output, 0, summary)

    def test_failure_is_preserved_on_successful_retry(self):
        inputs = self.load()
        summary = pilot.initialize_run(self.config, inputs, self.output)
        with patch.object(pilot, 'execute_fit', side_effect=TimeoutError('test limit')):
            self.assertEqual(pilot.run_task(self.config, inputs, self.output, 0, summary), 2)
        failed = self.output / 'tasks/task_000/status.json'
        self.assertEqual(json.loads(failed.read_text())['status'], 'timeout')
        self.assertEqual(pilot.run_task(self.config, inputs, self.output, 0, summary), 0)
        archived = self.output / 'incomplete/task_000_attempt_001/status.json'
        self.assertEqual(json.loads(archived.read_text())['status'], 'timeout')
        self.assertEqual(json.loads(failed.read_text())['status'], 'complete')

    def test_run_configuration_cannot_change_in_place(self):
        inputs = self.load()
        first = pilot.initialize_run(self.config, inputs, self.output)
        self.assertEqual(first, pilot.initialize_run(self.config, inputs, self.output))
        self.config['sampling']['seed'] += 1
        changed = self.load()
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            pilot.initialize_run(self.config, changed, self.output)

    def test_foreign_output_and_overlarge_grid_rejected(self):
        self.output.mkdir()
        (self.output / 'existing.txt').write_text('preserve')
        with self.assertRaises(FileExistsError):
            pilot.initialize_run(self.config, self.load(), self.output)
        self.assertEqual((self.output / 'existing.txt').read_text(), 'preserve')
        self.config['models'].update(k=[1, 2, 3], seeds=[1, 2, 3], density_neighbors=[1, 2], density_quantiles=[.2, .5])
        with self.assertRaisesRegex(ValueError, 'at most 72'):
            pilot.validate_config(self.config)

    def test_task_lock_prevents_concurrent_mutation(self):
        inputs = self.load()
        summary = pilot.initialize_run(self.config, inputs, self.output)
        with pilot.file_lock(self.output / 'locks/task_000.lock'):
            with self.assertRaisesRegex(RuntimeError, 'holds the lock'):
                pilot.run_task(self.config, inputs, self.output, 0, summary)

    def test_preflight_authenticates_vectors_and_detects_modified_shard(self):
        config_path = self.root / 'config.json'
        pilot.write_json(config_path, self.config)
        self.assertEqual(pilot.main(['--config', str(config_path), '--data-root', str(self.root), '--preflight']), 0)
        with (self.root / 'shards/one.npz').open('ab') as stream:
            stream.write(b'corrupt')
        self.assertEqual(pilot.main(['--config', str(config_path), '--data-root', str(self.root), '--preflight']), 2)

    def test_centrality_ignores_rejected_and_uses_within_region_unit_mean(self):
        X = np.asarray([[1., 0], [0., 1], [-1., 0]])
        result = pilot.centrality(X, np.asarray([0, 0, -1]))
        np.testing.assert_allclose(result[:2], [2**-.5, 2**-.5], rtol=1e-6)
        self.assertTrue(np.isnan(result[2]))

    def test_full_grid_aggregates_and_distinguishes_independent_from_nested(self):
        inputs = self.load()
        summary = pilot.initialize_run(self.config, inputs, self.output)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.aggregate_run(self.config, inputs, self.output, summary), 2)
            for task in pilot.planned_tasks(self.config):
                self.assertEqual(pilot.run_task(self.config, inputs, self.output, task['task_id'], summary), 0)
            self.assertEqual(pilot.aggregate_run(self.config, inputs, self.output, summary), 0)
        result = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['completed_fits'], 18)
        self.assertEqual(result['comparisons']['independent_fit_reproducibility'], 9)
        self.assertEqual(result['comparisons']['nested_training_size_consistency'], 18)
        self.assertEqual(result['human_interpretability_assessment'], 'pending_manual_review')
        self.assertFalse(result['availability_effects_estimated'])


if __name__ == '__main__':
    unittest.main()
