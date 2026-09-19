"""Offline integration checks for the deletion gate; no Azure or SSH requests."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_azure_semantic_pilot import embed, prepare
from test_azure_semantic_pilot import FakeClient, FakeTokenizer, prepared_fixture
from verify_azure_embeddings_complete import MODEL, PREPARED, sha256, verify_corpus


class Client(FakeClient):
    def embed(self, inputs):
        result = super().embed(inputs)
        result['model'] = MODEL
        return result


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            prepared_fixture(self.root)
            self.prepared = self.root / PREPARED
            prepare(self.root / 'source_config.json', self.root / 'texts.csv', self.prepared,
                    self.root, tokenizer=FakeTokenizer(), all_posts=True)
            (self.root / 'code').mkdir()
            for name in ('build_azure_semantic_pilot.py', 'unc_azure_embeddings.py'):
                shutil.copyfile(Path(__file__).resolve().parents[1] / name, self.root / 'code' / name)
            with self.assertRaises(RuntimeError):
                embed(self.prepared, MODEL, self.root, client=Client(fail_after=2))
            files = {}
            for directory in (self.prepared, self.root / 'code'):
                for path in directory.rglob('*'):
                    if path.is_file():
                        files[str(path.relative_to(self.root))] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
            manifest = {'total_posts': 240, 'file_count': len(files), 'files': files}
            (self.root / 'transfer_manifest.json').write_text(json.dumps(manifest))
            self.pin = sha256(self.root / 'transfer_manifest.json')
            embed(self.prepared, MODEL, self.root, client=Client())
        self.cache = next(self.prepared.glob('azure_cache_*'))

    def verify(self):
        return verify_corpus(self.root, self.pin, expected_posts=240, expected_dimension=8)

    def test_completed_resume_passes(self):
        result = self.verify()
        self.assertTrue(result['complete'])
        self.assertEqual((result['posts'], result['dimension'], result['request_batches']), (240, 8, 8))

    def test_manifest_pin_is_required(self):
        with self.assertRaisesRegex(ValueError, 'source manifest changed'):
            verify_corpus(self.root, '0' * 64, expected_posts=240, expected_dimension=8)

    def test_initial_cached_vectors_cannot_change(self):
        with (self.cache / 'batch_00000.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Original transferred file changed'):
            self.verify()

    def test_incomplete_and_wrong_dimension_rejected(self):
        path = self.prepared / 'embedding_status.json'
        original = json.loads(path.read_text())
        path.write_text(json.dumps({**original, 'status': 'running'}))
        with self.assertRaisesRegex(ValueError, 'has not completed'):
            self.verify()
        path.write_text(json.dumps({**original, 'dimension': 7}))
        with self.assertRaisesRegex(ValueError, 'wrong corpus, dimension, or model'):
            self.verify()

    def test_final_vectors_checked_beyond_manifest_hash(self):
        manifest_path = self.cache / 'embedding_cache_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        name = next(iter(manifest['completed_shards']))
        path = self.cache / 'post_embedding_shards' / name
        with np.load(path, allow_pickle=False) as data:
            fields = dict(data)
        fields['embeddings'][0] *= 2
        np.savez(path, **fields)
        manifest['completed_shards'][name] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'unnormalized final vectors'):
            self.verify()

    def test_missing_new_batch_rejected(self):
        (self.cache / 'batch_00007.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing verified file'):
            self.verify()

    def test_active_writer_blocks_verification(self):
        with (self.prepared / '.embedding.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.verify()

    def test_symlinked_final_index_rejected(self):
        path = self.cache / 'post_embedding_index.csv'
        copy = self.root / 'copied_index.csv'
        path.rename(copy)
        path.symlink_to(copy)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
