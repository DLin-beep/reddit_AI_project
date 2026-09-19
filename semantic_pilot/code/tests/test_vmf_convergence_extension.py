import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_semantic_threefold_pilot as pilot
import run_vmf_convergence_extension as extension
from semantic_region_directional import fit_vmf
from semantic_vmf_continuation import continue_vmf
import test_semantic_threefold_pilot as fixture_module


class ConvergenceExtensionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ThreefoldPilotTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.config = self.fixture.root, self.fixture.config
        self.config['models'].update(families=['spherical_kmeans', 'vmf'], max_iter=100, tolerance=1e-7)
        # Broad, overlapping content ensures a short original vMF fit is unfinished.
        shard = self.root / 'shards/one.npz'
        with np.load(shard, allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        rng = np.random.default_rng(99)
        vectors = rng.normal(size=(24, 3)) + np.asarray([.3, .1, 0.])
        values['embeddings'] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        np.savez_compressed(shard, **values)
        manifest = extension.read_json(self.root / 'manifest.json')
        manifest['completed_shards']['one.npz'] = pilot.sha256(shard)
        pilot.write_json(self.root / 'manifest.json', manifest)
        self.config['input_sha256']['manifest'] = pilot.sha256(self.root / 'manifest.json')
        self.source, self.output = self.root / 'original', self.root / 'extended'
        self.inputs = pilot.load_inputs(self.config, self.root)
        summary = pilot.initialize_run(self.config, self.inputs, self.source)
        def short_initial(X, k, seed, **kwargs):
            kwargs['max_iter'] = 1
            return fit_vmf(X, k, seed, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(pilot, 'fit_vmf', side_effect=short_initial):
            for spec in pilot.planned_tasks(self.config):
                self.assertEqual(pilot.run_task(self.config, self.inputs, self.source, spec['task_id'], summary), 0)
        self.original_hashes = {str(path.relative_to(self.source)): pilot.sha256(path)
                                for path in self.source.rglob('*') if path.is_file()}
        extension.prepare(self.source, self.root, self.output, chunk_iterations=2, checkpoint_every=1)
        self.pending = extension.read_json(self.output / 'extension.json')['pending_task_ids']
        self.assertTrue(self.pending)

    def test_prepare_preserves_numerical_artifacts_and_original_results(self):
        metadata = extension.read_json(self.output / 'extension.json')
        for task_id in metadata['carried_forward_task_ids']:
            original, revised = extension.task_path(self.source, task_id), extension.task_path(self.output, task_id)
            for name in ('predictions.npz', 'review_assignments_restricted.csv'):
                self.assertEqual(pilot.sha256(original / name), pilot.sha256(revised / name))
            with np.load(original / 'model.npz', allow_pickle=False) as before, np.load(revised / 'model.npz', allow_pickle=False) as after:
                for key in before.files:
                    if key != 'input_fingerprint':
                        np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(self.original_hashes, {str(path.relative_to(self.source)): pilot.sha256(path)
                                              for path in self.source.rglob('*') if path.is_file()})
        self.assertIsNone(extension.read_json(self.output / 'config.json')['optimization']['total_em_iteration_limit'])
        # Preparing again is a read-only resume operation.
        extension.prepare(self.source, self.root, self.output, chunk_iterations=2, checkpoint_every=1)

    def test_checkpoint_pause_resume_matches_uninterrupted_fit(self):
        task_id = self.pending[0]
        original = extension.load_model(extension.task_path(self.source, task_id))
        def pause_after_one(X, model, **kwargs):
            kwargs['chunk_iterations'] = 1
            answer = continue_vmf(X, model, **kwargs)
            if not answer['converged']:
                answer['stop_reason'] = 'stop_requested'
            return answer
        with contextlib.redirect_stdout(io.StringIO()), patch.object(extension, 'continue_vmf', side_effect=pause_after_one):
            self.assertEqual(extension.run_task(self.source, self.root, self.output, task_id), 75)
        directory = extension.task_path(self.output, task_id)
        self.assertEqual(extension.read_json(directory / 'status.json')['status'], 'incomplete')
        checkpoint = extension.read_json(self.output / 'checkpoints' / f'task_{task_id:03d}' / 'checkpoint.json')
        self.assertEqual(checkpoint['model_scalars']['iterations'], original['iterations'] + 1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(extension.run_task(self.source, self.root, self.output, task_id), 0)
            self.assertEqual(extension.run_task(self.source, self.root, self.output, task_id), 0)
        finished = extension.load_model(directory)
        spec = pilot.planned_tasks(self.config)[task_id]
        train = [aid for fold in spec['train_folds'] for aid in self.inputs['folds'][fold]]
        vectors, _ = pilot.load_vectors(self.inputs['index'], train, self.inputs['manifest'], self.inputs['shard_dir'], 3)
        X = np.asarray([vectors[aid] for aid in train])
        uninterrupted = continue_vmf(X, original, chunk_iterations=10000, tol=self.config['models']['tolerance'])
        self.assertTrue(uninterrupted['converged'])
        self.assertEqual(finished['iterations'], uninterrupted['iterations'])
        for key in ('centers', 'weights', 'kappas', 'labels', 'log_likelihood_history'):
            np.testing.assert_allclose(finished[key], uninterrupted[key], rtol=1e-11, atol=1e-11)

    def test_wall_deadline_never_marks_unfinished_fit_complete(self):
        task_id = self.pending[0]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(extension.run_task(self.source, self.root, self.output, task_id, wall_seconds=1e-12), 75)
            self.assertEqual(extension.aggregate(self.source, self.root, self.output), 75)
        status = extension.read_json(extension.task_path(self.output, task_id) / 'status.json')
        self.assertEqual(status['status'], 'incomplete')
        self.assertFalse(status['converged'])
        self.assertFalse((self.output / 'report_status.json').exists())

    def test_source_code_and_checkpoint_corruption_are_rejected(self):
        with patch.object(extension, 'source_hashes', return_value={'modified.py': 'a'}):
            with self.assertRaisesRegex(ValueError, 'source code changed'):
                extension.load_run(self.source, self.root, self.output)
        task_id = self.pending[0]
        with contextlib.redirect_stdout(io.StringIO()):
            extension.run_task(self.source, self.root, self.output, task_id, wall_seconds=1e-12)
        checkpoint_dir = self.output / 'checkpoints' / f'task_{task_id:03d}'
        checkpoint = extension.read_json(checkpoint_dir / 'checkpoint.json')
        with (checkpoint_dir / checkpoint['model_file']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'payload hash mismatch'):
            extension.run_task(self.source, self.root, self.output, task_id)

    def test_full_revision_aggregate_authenticates_all_outputs_and_counts(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for task_id in self.pending:
                self.assertEqual(extension.run_task(self.source, self.root, self.output, task_id), 0)
            self.assertEqual(extension.aggregate(self.source, self.root, self.output), 0)
        result = extension.read_json(self.output / 'report_status.json')
        self.assertEqual(result['completed_fits'], 12)
        self.assertEqual(result['converged_vmf_fits'], 6)
        self.assertEqual(result['fit_family_counts'], {'spherical_kmeans': 6, 'vmf': 6})
        self.assertEqual(result['comparisons'], {'independent_fit_reproducibility': 6, 'nested_training_size_consistency': 12})
        self.assertEqual(result, extension.read_json(self.output / 'status.json'))
        for name, expected in result['output_sha256'].items():
            self.assertEqual(pilot.sha256(self.output / name), expected, name)
        self.assertIn('vmf_convergence_summary.csv', result['output_sha256'])
        self.assertIn('no total EM iteration limit', (self.output / 'RESULTS.md').read_text())
        self.assertEqual(self.original_hashes, {str(path.relative_to(self.source)): pilot.sha256(path)
                                              for path in self.source.rglob('*') if path.is_file()})


if __name__ == '__main__':
    unittest.main()
