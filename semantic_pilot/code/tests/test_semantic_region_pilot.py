"""End-to-end checks on synthetic inputs only; no research-data model fits."""
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_semantic_region_pilot import (align_labels, load_metadata, load_vectors,
                                      model_specs, partition_comparison, run,
                                      sha256, write_csv, write_json)


def fixture(root):
    rng = np.random.default_rng(402)
    dimension = 16
    rows, roles, vectors = [], [], []
    for community in range(4):
        for role in ('training', 'heldout_historical'):
            for j in range(30):
                aid = f'post{len(rows):04d}'
                label = j % 3
                x = np.eye(dimension)[label] + rng.normal(0, .08, dimension)
                x /= np.linalg.norm(x)
                rows.append(dict(annotation_id=aid, post_id=aid, subreddit=f'c{community}',
                                 year_month='2022-01', text_sha256=f'text_hash_{aid}',
                                 shard='posts_000.npz', shard_row=len(rows)))
                roles.append(dict(annotation_id=aid, sample_role=role))
                vectors.append(x)
    reviews = [dict(annotation_id=r['annotation_id'], post_id=r['post_id'],
                    text_sha256=r['text_sha256'], derek_personal_req_score=str(i % 4),
                    arya_personal_req_score=str((i + 1) % 4))
               for i, r in enumerate(rows[::15])]
    write_csv(root / 'index.csv', rows)
    write_csv(root / 'roles.csv', roles)
    write_csv(root / 'reviews.csv', reviews)
    shards = root / 'shards'
    shards.mkdir()
    np.savez(shards / 'posts_000.npz', cache_key=np.asarray('synthetic_only'),
             annotation_ids=np.asarray([r['annotation_id'] for r in rows]),
             text_sha256=np.asarray([r['text_sha256'] for r in rows]),
             embeddings=np.asarray(vectors, dtype=np.float32))
    write_json(root / 'manifest.json', dict(cache_key='synthetic_only',
                completed_shards={'posts_000.npz': sha256(shards / 'posts_000.npz')}))
    return {'input': {'index': 'index.csv', 'roles': 'roles.csv', 'human_reviews': 'reviews.csv',
                      'manifest': 'manifest.json', 'shards': 'shards', 'dimension': dimension},
            'sampling': {'seed': 42, 'training_per_community': 20, 'evaluation_per_community': 20,
                         'stability_training_fraction': .8},
            'models': {'k': [3], 'seeds': [11, 29], 'max_iter': 70, 'tolerance': 1e-5,
                       'vmf_kappa_max': 300, 'density_neighbors': [5],
                       'density_quantiles': [0, .5], 'density_min_region_size': 3},
            'resources': {'threads': 1, 'fit_wall_limit_seconds': 60}}


class PilotTests(unittest.TestCase):
    def test_end_to_end_reuses_scores_and_preserves_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            before = sha256(root / 'reviews.csv')
            output = root / 'result'
            self.assertEqual(run(config, root, root / 'shards', output), 0)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['completed_fits'], len(model_specs(config)))
            self.assertEqual(status['new_human_scores'], 0)
            self.assertEqual(sha256(root / 'reviews.csv'), before)
            with (root / 'reviews.csv').open(newline='') as f:
                original = {r['annotation_id']: r for r in csv.DictReader(f)}
            with (output / 'review_assignments_restricted.csv').open(newline='') as f:
                assignments = list(csv.DictReader(f))
            self.assertEqual(len(assignments), len(original) * len(model_specs(config)))
            for r in assignments:
                self.assertEqual(r['derek_personal_req_score'], original[r['annotation_id']]['derek_personal_req_score'])
                self.assertEqual(r['arya_personal_req_score'], original[r['annotation_id']]['arya_personal_req_score'])
                if r['historical_role'] == 'heldout_historical':
                    self.assertEqual(r['used_in_this_fit'], '0')
            with (output / 'selected_posts_restricted.csv').open(newline='') as f:
                selected = list(csv.DictReader(f))
            self.assertEqual(sum(int(r['pilot_training']) for r in selected), 80)
            self.assertEqual(sum(int(r['pilot_evaluation']) for r in selected), 80)
            self.assertFalse(any(r['pilot_training'] == r['pilot_evaluation'] == '1' for r in selected))
            for file in (output / 'fits').glob('vmf*.npz'):
                with np.load(file, allow_pickle=False) as fit:
                    np.testing.assert_allclose(fit['evaluation_memberships'].sum(axis=1), 1, atol=1e-6)
                    self.assertLessEqual(fit['kappas'].max(), config['models']['vmf_kappa_max'])

    def test_corrupt_shard_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            index, roles, reviews, manifest, selected, paths = load_metadata(config, root)
            with (root / 'shards/posts_000.npz').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Shard hash mismatch'):
                load_vectors(index, selected['training'], manifest, root / 'shards', 16)

    def test_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            text = (root / 'reviews.csv').read_text().replace('text_hash_post0000', 'wrong_hash')
            (root / 'reviews.csv').write_text(text)
            with self.assertRaisesRegex(ValueError, 'identity/text hash'):
                load_metadata(config, root)

    def test_missing_community_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            with (root / 'roles.csv').open(newline='') as f:
                roles = list(csv.DictReader(f))
            for row in roles[:60]:
                row['sample_role'] = 'training'
            write_csv(root / 'roles.csv', roles)
            with self.assertRaisesRegex(ValueError, 'lacks 20 posts'):
                load_metadata(config, root)

    def test_interrupted_fit_cannot_report_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            with patch('run_semantic_region_pilot.fit_spherical_kmeans', side_effect=KeyboardInterrupt):
                result = run(config, root, root / 'shards', root / 'result')
            self.assertEqual(result, 130)
            status = json.loads((root / 'result/status.json').read_text())
            self.assertEqual(status['status'], 'interrupted')
            self.assertEqual(status['completed_fits'], 0)

    def test_collapsed_or_rejected_outputs_do_not_get_perfect_stability(self):
        self.assertIsNone(partition_comparison(np.zeros(8), np.zeros(8))['adjusted_rand_joint_assigned'])
        result = partition_comparison(-np.ones(8), -np.ones(8))
        self.assertIsNone(result['adjusted_rand_joint_assigned'])
        self.assertEqual(result['joint_assignment_fraction'], 0)

    def test_alignment_does_not_hide_unmatched_or_rejected_labels(self):
        mapped = align_labels(np.asarray([0, 0, 1, 1, -1]),
                              np.asarray([3, 3, 4, 4, 7]), np.asarray([3, 4, 7, -1]))
        np.testing.assert_array_equal(mapped, [0, 1, -2, -1])


if __name__ == '__main__':
    unittest.main()
