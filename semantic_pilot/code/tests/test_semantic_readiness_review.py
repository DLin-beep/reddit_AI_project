"""Synthetic checks for the bounded, blinded semantic interpretation packet."""

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_semantic_readiness_review as packet
import run_semantic_threefold_pilot as pilot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_specs():
    result = []
    for family in ('spherical_kmeans', 'vmf'):
        for k in (10, 20):
            base = f'{family}_k{k}'
            result.append(dict(task_id=len(result), fit_id=f'{base}_seed11_trainAB',
                               family=family, base=base, k=k, seed=11,
                               train_folds=['A', 'B'], eval_folds=['C']))
    return result


def candidate_tasks():
    tasks = []
    ids = np.array([f'post_{i:03d}' for i in range(60)])
    labels = np.array([0] * 40 + [1] * 7 + [2] * 13, dtype=np.int64)
    for spec in candidate_specs():
        tasks.append(dict(spec=spec, fit={'model_scalars': {'converged': True}},
                          predictions={'evaluation_ids_C': ids.copy(),
                                       'evaluation_labels_C': labels.copy(),
                                       'evaluation_centrality_C': np.arange(60, dtype=float),
                                       'evaluation_confidence_C': np.arange(60, dtype=float) / 60},
                          model={}, reviews=[]))
    return tasks


class SelectionTests(unittest.TestCase):
    def test_four_candidates_are_recorded_settings_not_selected_for_outcomes(self):
        specs = candidate_specs()
        distractors = []
        for number, changes in enumerate(({'seed': 29}, {'train_folds': ['A'], 'eval_folds': ['B', 'C']},
                                          {'family': 'density', 'k': None}, {'k': 30})):
            distractors.append({**specs[0], **changes, 'task_id': 10 + number,
                                'fit_id': f'distractor_{number}'})
        selected = packet.select_candidate_specs(list(reversed(specs + distractors)))
        self.assertEqual(selected, specs)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(spec['seed'] == 11 and spec['train_folds'] == ['A', 'B']
                            and spec['eval_folds'] == ['C'] for spec in selected))

    def test_missing_or_duplicate_candidate_is_an_error(self):
        specs = candidate_specs()
        with self.assertRaises(ValueError):
            packet.select_candidate_specs(specs[:-1])
        with self.assertRaises(ValueError):
            packet.select_candidate_specs(specs + [{**specs[0], 'task_id': 99}])

    def test_25_uniform_hash_ranked_posts_per_region_or_all_if_smaller(self):
        tasks = candidate_tasks()
        selected, maps, regions = packet.select_region_samples(tasks, seed=19, per_region=25)
        self.assertEqual(len(maps), 4)
        self.assertEqual(len(regions), 60)
        self.assertEqual(sum(not row['evaluation_region_empty'] for row in regions), 12)
        self.assertEqual(len(selected), 4 * (25 + 7 + 13))
        counts = Counter((row['map_alias'], row['region_alias']) for row in selected)
        self.assertEqual(sorted(counts.values()), sorted([25, 7, 13] * 4))
        ids = [f'post_{i:03d}' for i in range(40)]
        expected = set(sorted(ids, key=lambda aid: hashlib.sha256(f'19|{aid}'.encode()).hexdigest())[:25])
        for map_alias in {row['map_alias'] for row in selected}:
            map_rows = [row for row in selected if row['map_alias'] == map_alias]
            self.assertEqual(len({row['annotation_id'] for row in map_rows}), len(map_rows))
            actual_large = {row['annotation_id'] for row in map_rows if int(row['annotation_id'][-3:]) < 40}
            self.assertEqual(actual_large, expected)
            self.assertEqual({row['annotation_id'] for row in map_rows if int(row['annotation_id'][-3:]) >= 40},
                             {f'post_{i:03d}' for i in range(40, 60)})

    def test_order_changes_and_scores_do_not_change_selected_posts(self):
        tasks = candidate_tasks()
        first = packet.select_region_samples(tasks, seed=19, per_region=25)
        changed = deepcopy(tasks)
        for task in changed:
            for key, values in task['predictions'].items():
                task['predictions'][key] = values[::-1]
            task['predictions']['evaluation_centrality_C'][:] = -999
            task['predictions']['evaluation_confidence_C'][:] = 1
            task['reviews'] = [{'annotation_id': f'post_{i:03d}',
                                'derek_personal_req_score': '3', 'arya_personal_req_score': '0',
                                'region': 0} for i in range(60)]
            task['fit']['stability'] = 12345
        second = packet.select_region_samples(changed[::-1], seed=19, per_region=25)
        self.assertEqual(first, second)

    def test_duplicate_mismatched_or_invalid_evaluation_ids_are_rejected(self):
        for kind in ('duplicate', 'different_universe', 'wrong_shape', 'noninteger_labels'):
            with self.subTest(kind=kind):
                tasks = candidate_tasks()
                prediction = tasks[0]['predictions']
                if kind == 'duplicate':
                    prediction['evaluation_ids_C'][1] = prediction['evaluation_ids_C'][0]
                elif kind == 'different_universe':
                    prediction['evaluation_ids_C'][1] = 'unknown'
                elif kind == 'wrong_shape':
                    prediction['evaluation_labels_C'] = prediction['evaluation_labels_C'][:-1]
                else:
                    prediction['evaluation_labels_C'] = prediction['evaluation_labels_C'].astype(float) + .1
                with self.assertRaises(ValueError):
                    packet.select_region_samples(tasks)


class AuthenticatedPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_root = self.root / 'data'
        self.data_root.mkdir()
        self.results = self.root / 'experiment' / 'results'
        self.results.mkdir(parents=True)
        self.output = self.root / 'review_packet'
        self.config = {
            'input': {'index': 'index.csv', 'manifest': 'manifest.json', 'roles': 'roles.csv',
                      'human_reviews': 'reviews.csv', 'texts': 'texts.csv',
                      'shards': 'shards', 'dimension': 3},
            'sampling': {'seed': 20260914, 'expected_posts': 180, 'expected_communities': 1,
                         'expected_posts_per_community': 180, 'expected_human_reviews': 2},
            'models': {'families': ['spherical_kmeans', 'vmf'], 'k': [10, 20], 'seeds': [11],
                       'density_neighbors': [15], 'density_quantiles': [.5], 'density_min_region_size': 30,
                       'max_iter': 100, 'tolerance': 1e-5, 'vmf_kappa_max': 4000},
            'resources': {'threads': 1, 'fit_wall_limit_seconds': 60},
        }
        rows, text_rows = [], []
        for i in range(180):
            aid, post_id = f'annotation_{i:03d}', f'physical_{i:03d}'
            text = f'Plain synthetic content {i}. <script>alert("untrusted")</script> & \"quoted\"'
            text_rows.append({'annotation_id': aid, 'post_id': post_id, 'text': text})
            rows.append(dict(annotation_id=aid, post_id=post_id, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             subreddit='synthetic_community', year_month='2020-01', shard='one.npz', shard_row=i))
        write_csv(self.data_root / 'index.csv', rows)
        write_csv(self.data_root / 'texts.csv', text_rows)
        write_csv(self.data_root / 'roles.csv', [dict(annotation_id=row['annotation_id'], post_id=row['post_id'],
                    text_sha256=row['text_sha256'], sample_role='training') for row in rows])
        folds = pilot.make_folds({row['annotation_id']: row for row in rows}, self.config['sampling'])
        self.review_ids = folds['C'][:2]
        review_rows = []
        for row in rows:
            if row['annotation_id'] in self.review_ids:
                review_rows.append(dict(annotation_id=row['annotation_id'], post_id=row['post_id'],
                                   text_sha256=row['text_sha256'], derek_personal_req_score='3',
                                   arya_personal_req_score='1'))
        write_csv(self.data_root / 'reviews.csv', review_rows)
        (self.data_root / 'shards').mkdir()
        # The packet only reads authenticated text and assignments, not vectors.
        # An empty pickle-free archive is enough for metadata-level existence.
        np.savez_compressed(self.data_root / 'shards/one.npz', unused=np.array([1]))
        write_json(self.data_root / 'manifest.json', {
            'records': 180, 'cache_key': 'synthetic_cache',
            'encoder': {'embedding_dimension': 3, 'normalize_embeddings': True},
            'completed_shards': {'one.npz': sha(self.data_root / 'shards/one.npz')},
        })
        self.config['input_sha256'] = {key: sha(self.data_root / self.config['input'][key])
                                      for key in ('index', 'manifest', 'roles', 'human_reviews', 'texts')}
        self.inputs = pilot.load_inputs(self.config, self.data_root)
        self.summary = pilot.input_summary(self.config, self.inputs)
        self.specs = candidate_specs()
        write_json(self.results / 'config.json', self.config)
        write_json(self.results / 'inputs.json', self.summary)
        write_json(self.results / 'tasks.json', self.specs)
        write_csv(self.results / 'fold_assignments_restricted.csv', pilot.fold_rows(self.inputs))
        code_copy = self.results.parent / 'code'
        code_copy.mkdir()
        for name in self.summary['code_sha256']:
            (code_copy / name).write_bytes((Path(pilot.__file__).parent / name).read_bytes())
        train_ids = folds['A'] + folds['B']
        eval_ids = folds['C']
        eval_labels = np.array([0] * 40 + [1] * 7 + [2] * 13, dtype=np.int64)
        for spec in self.specs:
            directory = self.results / 'tasks' / f"task_{spec['task_id']:03d}"
            directory.mkdir(parents=True)
            write_json(directory / 'fit.json', {
                'task': spec, 'input_fingerprint': self.summary['input_fingerprint'],
                'config_sha256': digest(self.config), 'model_scalars': {'converged': True, 'iterations': 42},
                'summaries': [{'evaluation_fold': 'C', 'evaluation_coverage': 1.,
                               'evaluation_regions': 3, 'converged': True, 'seconds': .1}],
            })
            np.savez_compressed(directory / 'model.npz', weights=np.full(spec['k'], 1 / spec['k']),
                                kappas=np.full(spec['k'], 10.))
            np.savez_compressed(directory / 'predictions.npz', training_ids=np.array(train_ids),
                                training_labels=np.arange(len(train_ids), dtype=np.int64) % spec['k'],
                                evaluation_ids_C=np.array(eval_ids), evaluation_labels_C=eval_labels,
                                evaluation_centrality_C=np.arange(60, dtype=float),
                                evaluation_confidence_C=np.linspace(0, 1, 60),
                                review_ids=np.array(self.review_ids), review_labels=eval_labels[:2],
                                review_used_in_fit=np.zeros(2, dtype=np.int64))
            write_csv(directory / 'review_assignments_restricted.csv', [
                {**self.inputs['reviews'][aid], 'region': 0, 'fold': 'C', 'used_in_fit': 0}
                for aid in self.review_ids])
            write_json(directory / 'status.json', {
                'status': 'complete', 'task_id': spec['task_id'], 'fit_id': spec['fit_id'],
                'input_fingerprint': self.summary['input_fingerprint'], 'config_sha256': digest(self.config),
                'artifact_sha256': {name: sha(directory / name) for name in
                                    ('fit.json', 'model.npz', 'predictions.npz', 'review_assignments_restricted.csv')},
            })
        write_json(self.results / 'report_status.json', {
            'status': 'complete', 'planned_fits': 4, 'completed_fits': 4, 'pending_fits': 0,
            'input_fingerprint': self.summary['input_fingerprint'], 'output_sha256': {},
        })

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return packet.build_packet(self.results, self.data_root, self.output, seed=19, per_region=25)

    def test_packet_does_not_modify_original_results_or_authenticated_sources(self):
        before = {str(path.relative_to(self.root)): sha(path)
                  for path in self.root.rglob('*') if path.is_file()}
        result = self.build()
        self.assertTrue(result)
        after = {str(path.relative_to(self.root)): sha(path)
                 for path in self.root.rglob('*') if path.is_file() and self.output not in path.parents}
        self.assertEqual(before, after)
        self.assertTrue(self.output.is_dir())

    def test_reviewer_view_hides_method_ratings_identity_and_stability(self):
        status = self.build()
        self.assertEqual(status['status'], 'prepared_for_manual_review')
        self.assertEqual(status['manual_review_status'], 'not_started')
        self.assertEqual(status['event_window_support'], 'unavailable')
        reader_dir = self.output / 'reviewer'
        rows = pilot.read_csv(reader_dir / 'sampled_posts_restricted.csv')
        self.assertEqual(set(rows[0]), {'map_alias', 'region_alias', 'post_alias', 'display_order', 'text'})
        self.assertEqual(len(rows), 4 * (25 + 7 + 13))
        for path in reader_dir.iterdir():
            if not path.is_file():
                continue
            text = path.read_text(encoding='utf-8')
            for excluded in ('spherical_kmeans', 'vmf_k', 'task_id', 'fit_id', 'annotation_id',
                             'derek_personal_req_score', 'arya_personal_req_score',
                             'adjusted_rand', 'best_jaccard', 'synthetic_community', '2020-01'):
                with self.subTest(file=path.name, excluded=excluded):
                    self.assertNotIn(excluded, text)
            for aid in self.inputs['index']:
                self.assertNotIn(aid, text)
                self.assertNotIn(self.inputs['index'][aid]['post_id'], text)
        notes = pilot.read_csv(reader_dir / 'region_notes.csv')
        self.assertEqual(len(notes), 60)
        self.assertTrue(all(not row['short_label'] and not row['what_this_region_captures']
                            and not row['coherence_and_consistency'] for row in notes))
        ratings = pilot.read_csv(self.output / 'restricted_key/existing_ratings_context.csv')
        self.assertEqual(len(ratings), 8)
        self.assertTrue(all(row['derek_personal_req_score'] == '3' and row['arya_personal_req_score'] == '1'
                            for row in ratings))
        manifest = json.loads((self.output / 'manifest.json').read_text())
        self.assertFalse(manifest['map_selected_or_refitted'])
        self.assertFalse(manifest['availability_effects_estimated'])
        self.assertEqual(manifest['new_human_scores'], 0)
        for relative, expected in manifest['output_sha256'].items():
            self.assertEqual(sha(self.output / relative), expected)

    def test_reader_escapes_post_markup_and_custom_display_labels(self):
        self.build()
        text = (self.output / 'reviewer/reader.html').read_text()
        self.assertNotIn('<script>', text)
        self.assertIn('&lt;script&gt;alert(&quot;untrusted&quot;)&lt;/script&gt;', text)
        self.assertIn('Content-Security-Policy', text)
        self.assertNotIn('<img ', text)
        malicious = '<img src=x onerror="alert(1)">'
        custom = packet.render_reader(
            [{'map_alias': malicious, 'region_alias': malicious, 'post_alias': malicious,
              'display_order': malicious, 'text': malicious}],
            [{'map_alias': malicious, 'region_alias': malicious}])
        self.assertNotIn(malicious, custom)
        self.assertNotIn('<img ', custom)
        self.assertIn('&lt;img src=x onerror=&quot;alert(1)&quot;&gt;', custom)

    def test_existing_packet_and_source_locations_cannot_be_overwritten(self):
        self.build()
        notes = self.output / 'reviewer/region_notes.csv'
        notes.write_text(notes.read_text() + '\nRecorded reviewer note\n')
        before = {str(path.relative_to(self.output)): sha(path)
                  for path in self.output.rglob('*') if path.is_file()}
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before, {str(path.relative_to(self.output)): sha(path)
                                 for path in self.output.rglob('*') if path.is_file()})
        for target in (self.results, self.results / 'new_packet', self.results.parent):
            with self.subTest(target=target), self.assertRaises(ValueError):
                packet.build_packet(self.results, self.data_root, target)

    def test_task_artifact_tamper_fails_before_packet_success(self):
        (self.results / 'tasks/task_000/fit.json').write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.output / 'manifest.json').exists())

    def test_authenticated_text_source_tamper_is_rejected(self):
        with (self.data_root / 'texts.csv').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaises(ValueError):
            self.build()

    def test_changed_recorded_code_or_fold_assignment_is_rejected(self):
        for kind in ('code', 'fold'):
            with self.subTest(kind=kind):
                target = (self.results.parent / 'code/run_semantic_threefold_pilot.py'
                          if kind == 'code' else self.results / 'fold_assignments_restricted.csv')
                original = target.read_bytes()
                if kind == 'code':
                    target.write_bytes(original + b'\n')
                else:
                    rows = pilot.read_csv(target)
                    rows[0]['fold'] = 'C' if rows[0]['fold'] != 'C' else 'A'
                    write_csv(target, rows)
                try:
                    with self.assertRaises(ValueError):
                        self.build()
                finally:
                    target.write_bytes(original)

    def test_incomplete_report_is_not_interpretation_ready(self):
        status = json.loads((self.results / 'report_status.json').read_text())
        status['status'] = 'partial'
        write_json(self.results / 'report_status.json', status)
        with self.assertRaises(ValueError):
            self.build()

    def test_selected_text_requires_exact_content_and_unique_complete_ids(self):
        selected = set(self.inputs['folds']['C'][:2])
        texts = packet.load_selected_texts(self.inputs, selected)
        self.assertEqual(set(texts), selected)
        self.assertTrue(all(isinstance(text, str) for text in texts.values()))
        original = (self.data_root / 'texts.csv').read_bytes()
        original_hash = self.inputs['input_sha256']['texts']
        for kind in ('duplicate', 'missing', 'changed_text', 'unknown_selection'):
            with self.subTest(kind=kind):
                rows = pilot.read_csv(self.data_root / 'texts.csv')
                position = next(i for i, row in enumerate(rows) if row['annotation_id'] in selected)
                selected_now = selected
                if kind == 'duplicate':
                    rows.append(dict(rows[position]))
                elif kind == 'missing':
                    rows.pop(position)
                elif kind == 'changed_text':
                    rows[position]['text'] = 'Unrecorded replacement text'
                else:
                    selected_now = selected | {'not_a_corpus_post'}
                write_csv(self.data_root / 'texts.csv', rows)
                # Authenticate the synthetic file anew so these checks exercise
                # the ID/content validation beyond the separate file-hash guard.
                self.inputs['input_sha256']['texts'] = sha(self.data_root / 'texts.csv')
                try:
                    with self.assertRaises(ValueError):
                        packet.load_selected_texts(self.inputs, selected_now)
                finally:
                    (self.data_root / 'texts.csv').write_bytes(original)
                    self.inputs['input_sha256']['texts'] = original_hash


if __name__ == '__main__':
    unittest.main()
