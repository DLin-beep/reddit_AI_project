"""Offline Azure preparation/cache integration; no real API requests."""
import csv
import fcntl
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_azure_semantic_pilot import embed, prepare, verify_prepared
from run_semantic_region_pilot import load_metadata, load_vectors, run, sha256, write_csv, write_json
from test_semantic_region_pilot import fixture


class FakeTokenizer:
    def encode(self, text, disallowed_special=()):
        return [ord(c) for c in text]


class FakeClient:
    def __init__(self, fail_after=None, dimension=8):
        self.calls = 0
        self.fail_after = fail_after
        self.dimension = dimension

    def embed(self, inputs):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError('Simulated interrupted connection')
        self.calls += 1
        vectors = []
        for tokens in inputs:
            seed = int(hashlib.sha256(bytes(tokens)).hexdigest()[:8], 16)
            x = np.random.default_rng(seed).normal(size=self.dimension)
            vectors.append(x / np.linalg.norm(x))
        return {'embeddings': np.asarray(vectors), 'model': 'test-azure-model',
                'usage': {'prompt_tokens': sum(map(len, inputs)), 'total_tokens': sum(map(len, inputs))}}


def prepared_fixture(root):
    config = fixture(root)
    with (root / 'index.csv').open(newline='') as f:
        index = list(csv.DictReader(f))
    texts = []
    for row in index:
        text = 'Text for ' + row['annotation_id']
        row['text_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        texts.append({'annotation_id': row['annotation_id'], 'post_id': row['post_id'], 'text': text})
    hashes = {row['annotation_id']: row['text_sha256'] for row in index}
    with (root / 'reviews.csv').open(newline='') as f:
        reviews = list(csv.DictReader(f))
    for row in reviews:
        row['text_sha256'] = hashes[row['annotation_id']]
    write_csv(root / 'index.csv', index)
    write_csv(root / 'reviews.csv', reviews)
    write_csv(root / 'texts.csv', texts)
    write_json(root / 'source_config.json', config)
    plan = prepare(root / 'source_config.json', root / 'texts.csv', root / 'prepared', root,
                   max_tokens=16, overlap=3, tokenizer=FakeTokenizer())
    return root / 'prepared', plan


class AzurePilotTests(unittest.TestCase):
    def test_request_timeout_forwarded_and_change_preserves_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, _ = prepared_fixture(root)
            with patch.dict('os.environ', {'UNC_AI_API_KEY': 'offline-test-key'}), \
                    patch('build_azure_semantic_pilot.AzureEmbeddingClient', return_value=FakeClient()) as client_class:
                first = embed(prepared, 'actual-deployment', root, request_timeout=120)
                self.assertEqual(client_class.call_args.kwargs['timeout'], 120)
                client_class.assert_called_once()
            cached = FakeClient(fail_after=0)
            second = embed(prepared, 'actual-deployment', root, client=cached, request_timeout=240)
            self.assertEqual(first, second)
            self.assertEqual(cached.calls, 0)
            self.assertEqual(len(list(prepared.glob('azure_cache_*'))), 1)

    def test_invalid_request_timeout_fails_before_loading_or_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(fail_after=0)
            for value in (0, -1, float('nan'), float('inf'), True, '120'):
                with self.subTest(timeout=value), self.assertRaisesRegex(ValueError, 'timeout'):
                    embed(root / 'missing', 'actual-deployment', root, client=client, request_timeout=value)
            self.assertEqual(client.calls, 0)
            self.assertEqual(list(root.iterdir()), [])

    def test_example_deployment_fails_before_input_loading_or_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(fail_after=0)
            with self.assertRaisesRegex(ValueError, 'still a placeholder'):
                embed(root / 'not_prepared', 'YOUR_LARGE_MODEL_DEPLOYMENT', root, client=client)
            self.assertEqual(client.calls, 0)
            self.assertEqual(list(root.iterdir()), [])

    def test_example_key_fails_before_client_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, _ = prepared_fixture(root)
            with patch.dict('os.environ', {'UNC_AI_API_KEY': '123'}), \
                    patch('build_azure_semantic_pilot.AzureEmbeddingClient') as client_class:
                with self.assertRaisesRegex(ValueError, 'still an example key'):
                    embed(prepared, 'actual-deployment', root)
                client_class.assert_not_called()
            status = json.loads((prepared / 'embedding_status.json').read_text())
            self.assertEqual(status['cached_batches'], 0)

    def test_prepare_selects_only_pilot_and_existing_reviews(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, plan = prepared_fixture(root)
            self.assertEqual(plan['training_posts'], 80)
            self.assertEqual(plan['comparison_posts'], 80)
            self.assertEqual(plan['reviewed_posts'], 16)
            self.assertLessEqual(plan['unique_posts'], 176)
            self.assertEqual(plan['embedding_scope'], 'pilot_and_reviewed_posts')
            self.assertEqual(plan['indexed_posts'], 240)
            self.assertEqual(plan['api_requests'], 0)
            self.assertEqual(plan['model'], None)
            self.assertEqual(verify_prepared(prepared), plan)

    def test_all_posts_covers_index_and_preserves_pilot_selection_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot, pilot_plan = prepared_fixture(root)
            full = root / 'prepared_all'
            plan = prepare(root / 'source_config.json', root / 'texts.csv', full, root,
                           max_tokens=16, overlap=3, tokenizer=FakeTokenizer(), all_posts=True)
            source = json.loads((root / 'source_config.json').read_text())
            index, _, _, _, original_selected, _ = load_metadata(source, root)
            with (full / 'selected_posts_restricted.csv').open(newline='') as stream:
                posts = list(csv.DictReader(stream))
            self.assertEqual({row['annotation_id'] for row in posts}, set(index))
            self.assertEqual(len(posts), len(index))
            self.assertEqual(plan['unique_posts'], 240)
            self.assertEqual(plan['embedding_scope'], 'all_indexed_posts')
            self.assertGreater(plan['unique_posts'], pilot_plan['unique_posts'])
            for name in ('post_metadata.csv', 'roles.csv', 'human_reviews.csv', 'comparison_template.json'):
                self.assertEqual(sha256(full / name), sha256(pilot / name))
            self.assertEqual(plan['api_requests'], 0)
            result = embed(full, 'test-deployment', root, client=FakeClient())
            self.assertEqual(result['posts'], len(index))
            config = json.loads((full / 'comparison_config.json').read_text())
            new_index, _, _, manifest, selected, _ = load_metadata(config, root)
            self.assertEqual(selected, original_selected)
            vectors, _ = load_vectors(new_index, sorted(index), manifest,
                                      root / config['input']['shards'], config['input']['dimension'])
            self.assertEqual(set(vectors), set(index))
            self.assertTrue(all(vector.shape == (8,) for vector in vectors.values()))

    def test_all_posts_rejects_missing_text_outside_the_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot, _ = prepared_fixture(root)
            with (pilot / 'selected_posts_restricted.csv').open(newline='') as stream:
                pilot_ids = {row['annotation_id'] for row in csv.DictReader(stream)}
            with (root / 'texts.csv').open(newline='') as stream:
                texts = list(csv.DictReader(stream))
            missing_id = next(row['annotation_id'] for row in texts if row['annotation_id'] not in pilot_ids)
            write_csv(root / 'incomplete_texts.csv', [row for row in texts if row['annotation_id'] != missing_id])
            full = root / 'prepared_all'
            with self.assertRaisesRegex(ValueError, 'does not cover every selected post'):
                prepare(root / 'source_config.json', root / 'incomplete_texts.csv', full, root,
                        tokenizer=FakeTokenizer(), all_posts=True)
            self.assertFalse(full.exists())

    def test_cached_resume_and_generated_inputs_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, plan = prepared_fixture(root)
            human_hash = sha256(prepared / 'human_reviews.csv')
            client = FakeClient()
            result = embed(prepared, 'test-deployment', root, client=client)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['posts'], plan['unique_posts'])
            self.assertEqual(result['dimension'], 8)
            self.assertGreater(client.calls, 0)
            second = FakeClient(fail_after=0)
            resumed = embed(prepared, 'test-deployment', root, client=second)
            self.assertEqual(resumed, result)
            self.assertEqual(second.calls, 0)
            config = json.loads((prepared / 'comparison_config.json').read_text())
            self.assertEqual(config['input']['dimension'], 8)
            self.assertEqual(config['models']['vmf_kappa_max'], 150)
            self.assertEqual(sha256(prepared / 'human_reviews.csv'), human_hash)
            self.assertEqual(run(config, root, root / config['input']['shards'], root / 'fits'), 0)
            with (root / 'fits/selected_posts_restricted.csv').open(newline='') as f:
                selected = list(csv.DictReader(f))
            self.assertEqual(sum(int(r['pilot_training']) for r in selected), 80)
            self.assertEqual(sum(int(r['pilot_evaluation']) for r in selected), 80)

    def test_interruption_retains_completed_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, plan = prepared_fixture(root)
            first = FakeClient(fail_after=2)
            with self.assertRaisesRegex(RuntimeError, 'Simulated'):
                embed(prepared, 'test-deployment', root, client=first)
            status = json.loads((prepared / 'embedding_status.json').read_text())
            self.assertEqual(status['status'], 'incomplete')
            self.assertEqual(status['cached_batches'], 2)
            second = FakeClient()
            result = embed(prepared, 'test-deployment', root, client=second)
            self.assertEqual(second.calls, result['request_batches_cached'] - 2)

    def test_corrupt_cached_batch_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, plan = prepared_fixture(root)
            embed(prepared, 'test-deployment', root, client=FakeClient())
            cache = next(prepared.glob('azure_cache_*/batch_00000.npz'))
            with cache.open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                embed(prepared, 'test-deployment', root, client=FakeClient(fail_after=0))

    def test_changed_model_uses_a_separate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, plan = prepared_fixture(root)
            embed(prepared, 'deployment-one', root, client=FakeClient())
            client = FakeClient()
            embed(prepared, 'deployment-two', root, client=client)
            self.assertGreater(client.calls, 0)
            self.assertEqual(len(list(prepared.glob('azure_cache_*'))), 2)

    def test_budget_and_input_integrity_fail_before_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, plan = prepared_fixture(root)
            client = FakeClient(fail_after=0)
            with self.assertRaisesRegex(ValueError, 'token limit'):
                embed(prepared, 'deployment', root, client=client, max_total_tokens=1)
            with (prepared / 'chunks_restricted.jsonl').open('a') as f:
                f.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'Prepared input changed'):
                embed(prepared, 'deployment', root, client=client)
            self.assertEqual(client.calls, 0)

    def test_shared_prepared_lock_prevents_concurrent_model_handoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, plan = prepared_fixture(root)
            client = FakeClient(fail_after=0)
            with (prepared / '.embedding.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    embed(prepared, 'other-deployment', root, client=client)
            self.assertEqual(client.calls, 0)


if __name__ == '__main__':
    unittest.main()
