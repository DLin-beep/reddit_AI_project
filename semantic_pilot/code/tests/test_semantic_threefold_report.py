import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_threefold_report import (compare_partitions, comparison_kind,
                                       report_run, select_examples)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PartitionTests(unittest.TestCase):
    def test_arbitrary_label_permutations_do_not_change_comparison(self):
        summary, regions, edges = compare_partitions(np.array([0, 0, 1, 1, -1]), np.array([8, 8, 5, 5, -1]))
        self.assertEqual(summary['adjusted_rand_joint_assigned'], 1.)
        self.assertEqual(summary['joint_assignment_fraction'], .8)
        self.assertEqual(summary['both_rejected_posts'], 1)
        self.assertTrue(all(row['best_jaccard'] == 1. for row in regions))
        self.assertEqual(sum(row['one_to_one_match'] for row in edges), 2)
        self.assertTrue(all(row['region'] >= 0 for row in regions))

    def test_rejected_members_remain_in_jaccard_denominator(self):
        summary, rows, edges = compare_partitions(np.array([0, 0, 0, 1, 1]), np.array([7, 7, -1, 8, 8]))
        target = next(row for row in rows if row['side'] == 'left' and row['region'] == 0)
        self.assertAlmostEqual(target['best_jaccard'], 2 / 3)
        self.assertAlmostEqual(target['fraction_rejected_by_other'], 1 / 3)
        self.assertEqual(summary['adjusted_rand_joint_assigned'], 1.)
        self.assertEqual(summary['left_only_assigned_posts'], 1)

    def test_splits_and_merges_exposed_bidirectionally(self):
        summary, rows, edges = compare_partitions(np.array([0, 0, 0, 0, 1, 1]), np.array([5, 5, 6, 6, 7, 7]))
        source = next(row for row in rows if row['side'] == 'left' and row['region'] == 0)
        target = next(row for row in rows if row['side'] == 'right' and row['region'] == 5)
        self.assertEqual(source['overlapping_other_regions'], 2)
        self.assertEqual(target['best_overlap_source_fraction'], 1.)
        self.assertEqual(target['best_overlap_target_fraction'], .5)
        self.assertEqual(len(edges), 3)
        self.assertEqual(len(json.loads(source['overlaps_json'])), 2)
        self.assertLess(summary['adjusted_rand_joint_assigned'], 1.)

    def test_all_rejected_and_single_partition_are_not_perfect_stability(self):
        for a, b in (([-1, -1], [-1, -1]), ([0, 0], [8, 8]), ([0, 1], [-1, -1])):
            summary, regions, _ = compare_partitions(np.array(a), np.array(b))
            self.assertIsNone(summary['adjusted_rand_joint_assigned'])
            self.assertIn('joint_partition_degenerate', summary['degenerate_flags'])
        summary, regions, _ = compare_partitions(np.array([-1, -1]), np.array([-2, -2]))
        self.assertEqual(regions, [])
        self.assertEqual(summary['both_rejected_posts'], 2)

    def test_bad_or_empty_assignments_fail(self):
        for a, b in (([], []), ([0], [0, 1]), ([.2], [1])):
            with self.assertRaises(ValueError):
                compare_partitions(np.array(a), np.array(b))

    def test_design_categories_are_distinct(self):
        a = dict(family='vmf', base='vmf_k2', seed=11, train_folds=['A'])
        self.assertEqual(comparison_kind(a, {**a, 'train_folds': ['B']}), 'independent_fit_reproducibility')
        self.assertEqual(comparison_kind(a, {**a, 'train_folds': ['A', 'B']}), 'nested_training_size_consistency')
        self.assertEqual(comparison_kind(a, {**a, 'seed': 29}), 'seed_stability')
        self.assertIsNone(comparison_kind(a, {**a, 'seed': 29, 'train_folds': ['B']}))
        self.assertIsNone(comparison_kind(a, {**a, 'base': 'vmf_k3'}))
        self.assertIsNone(comparison_kind({**a, 'family': 'density'}, {**a, 'family': 'density', 'seed': 29}))


class ExampleTests(unittest.TestCase):
    def test_examples_are_distinct_deterministic_and_scores_ranked(self):
        ids = np.array([f'p{i}' for i in range(20)])
        labels = np.array([0] * 19 + [-1])
        scores = np.arange(20, dtype=float)
        rows = select_examples(ids, labels, scores, scores, seed=19)
        repeated = select_examples(ids[::-1], labels[::-1], scores[::-1], scores[::-1], seed=19)
        self.assertEqual(rows, repeated)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(set(row['annotation_id'] for row in rows)), 8)
        self.assertNotIn('p19', [row['annotation_id'] for row in rows])
        by_type = {kind: [row for row in rows if row['selection'] == kind]
                   for kind in ('random', 'central', 'lower_confidence')}
        self.assertEqual([len(by_type[k]) for k in by_type], [3, 3, 2])
        self.assertEqual([row['cosine_centrality'] for row in by_type['central']],
                         sorted([row['cosine_centrality'] for row in by_type['central']], reverse=True))

    def test_small_regions_do_not_duplicate_posts_or_invent_confidence(self):
        rows = select_examples(np.array(['a', 'b']), np.array([0, 0]))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['selection'] == 'random' for row in rows))
        self.assertTrue(all(row['cosine_centrality'] is None for row in rows))


class ReportTests(unittest.TestCase):
    def fixture(self, root):
        config = {'models': {'density_min_region_size': 2}, 'sampling': {'seed': 13}}
        write_json(root / 'config.json', config)
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        posts = {}
        texts = {}
        for fold in ('A', 'B', 'C'):
            for j in range(6):
                aid = f'{fold}{j}'
                text = f'Original text {aid}'
                texts[aid] = text
                posts[aid] = {'annotation_id': aid, 'post_id': f'id{aid}', 'subreddit': 'example',
                              'year_month': '2020-01', 'fold': fold,
                              'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
        reviews = {'C0': {'annotation_id': 'C0', 'derek_personal_req_score': '3', 'arya_personal_req_score': '1'}}
        tasks = []
        for i, train in enumerate((['A'], ['B'], ['A', 'B'])):
            spec = dict(task_id=i, fit_id=f'fit{i}', family='vmf', base='vmf_k2', seed=11,
                        train_folds=train, eval_folds=['C'])
            tasks.append(spec)
            directory = root / 'tasks' / f'task_{i:03d}'
            directory.mkdir(parents=True)
            summary = {'evaluation_fold': 'C', 'evaluation_coverage': 1., 'evaluation_regions': 2,
                       'seconds': .2, 'converged': True, 'concentration_cap_hits': 0}
            write_json(directory / 'fit.json', {'task': spec, 'summaries': [summary],
                       'model_scalars': {}, 'input_fingerprint': 'inputhash', 'config_sha256': config_hash})
            train_ids = [aid for aid, row in posts.items() if row['fold'] in train]
            train_labels = np.array([0 if int(aid[1]) < 3 else 1 for aid in train_ids])
            np.savez_compressed(directory / 'model.npz', weights=np.array([.5, .5]), kappas=np.array([2., 3.]))
            np.savez_compressed(directory / 'predictions.npz', training_ids=np.array(train_ids), training_labels=train_labels,
                                evaluation_ids_C=np.array([f'C{j}' for j in range(6)]),
                                evaluation_labels_C=np.array([0, 0, 0, 1, 1, 1]),
                                evaluation_centrality_C=np.array([1., .9, .8, .8, .9, 1.]),
                                evaluation_confidence_C=np.array([.8] * 6), review_ids=np.array(['C0']),
                                review_labels=np.array([0]), review_used_in_fit=np.array([0]))
            write_csv(directory / 'review_assignments_restricted.csv', [{**reviews['C0'], 'region': 0, 'fold': 'C', 'used_in_fit': 0}])
            artifacts = ['fit.json', 'model.npz', 'predictions.npz', 'review_assignments_restricted.csv']
            write_json(directory / 'status.json', {'status': 'complete', 'task_id': i, 'fit_id': f'fit{i}',
                       'input_fingerprint': 'inputhash', 'config_sha256': config_hash,
                       'artifact_sha256': {name: sha(directory / name) for name in artifacts}})
        write_json(root / 'tasks.json', tasks)
        write_csv(root / 'fold_assignments_restricted.csv', list(posts.values()))
        return posts, reviews, texts

    def test_integrated_report_distinguishes_comparisons_and_keeps_ratings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            posts, reviews, texts = self.fixture(root)
            result = report_run(root, posts, reviews, texts)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['comparisons'], {'independent_fit_reproducibility': 1, 'nested_training_size_consistency': 2})
            self.assertEqual(result['new_human_scores'], 0)
            self.assertEqual(result['event_window_support'], 'unavailable')
            self.assertEqual(result['authenticated_text_example_rows'], 18)
            with (root / 'region_review_summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            rated = [row for row in rows if int(row['reviewed_posts'])]
            self.assertTrue(all(row['derek_reviewed_mean'] == '3.0' and row['arya_reviewed_mean'] == '1.0' for row in rated))
            self.assertTrue(all(row['scope'] == 'historically_selected_reviewed_posts_only' for row in rated))
            self.assertIn('cannot establish independent reproducibility', (root / 'RESULTS.md').read_text())
            with (root / 'region_sizes.csv').open() as stream:
                sizes = list(csv.DictReader(stream))
            self.assertTrue(all(row['event_window_posts'] == '' for row in sizes))
            self.assertTrue(all(float(row['vmf_effective_training_mass']) > 0 for row in sizes))

    def test_bad_original_text_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            posts, reviews, texts = self.fixture(root)
            texts['C0'] = 'Modified text'
            with self.assertRaisesRegex(ValueError, 'text hash mismatch'):
                report_run(root, posts, reviews, texts)

    def test_changed_artifact_fails_authentication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / 'tasks/task_000/fit.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Artifact hash mismatch'):
                report_run(root)

    def test_missing_tasks_are_partial_not_success_or_zero_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            status = root / 'tasks/task_002/status.json'
            write_json(status, {'status': 'failed'})
            result = report_run(root)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['pending_fits'], 1)
            self.assertEqual(result['comparisons'], {'independent_fit_reproducibility': 1})
            self.assertEqual(result['authenticated_text_example_rows'], 0)
            self.assertFalse(result['availability_effects_estimated'])
            self.assertIn('Partial run', (root / 'RESULTS.md').read_text())

    def test_tampered_review_values_fail_original_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            posts, reviews, texts = self.fixture(root)
            reviews['C0']['derek_personal_req_score'] = '0'
            with self.assertRaisesRegex(ValueError, 'differs from original'):
                report_run(root, posts, reviews, texts)


if __name__ == '__main__':
    unittest.main()
