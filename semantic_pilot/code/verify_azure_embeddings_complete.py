#!/usr/bin/env python3
"""Verify the completed 60,000-post Azure corpus before deleting its local cache.

Read-only; supports execution via SSH stdin. Requires NumPy on Longleaf.
"""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

PREPARED = 'output/azure_semantic_corpus_20260914_inputs'
MODEL = 'text-embedding-3-large'
ENDPOINT = 'https://azureaiapi.cloud.unc.edu/openai/v1/embeddings'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and relative != '', 'Missing relative file path')
    part = PurePosixPath(relative)
    require(not part.is_absolute() and '..' not in part.parts and '\\' not in relative,
            'Unsafe relative file path')
    path = root
    for component in part.parts:
        path = path / component
        require(not path.is_symlink(), 'Symlink in verified file path')
    require(path.is_file(), 'Missing verified file: ' + relative)
    return path


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {row['annotation_id']: row for row in rows}
    require(len(result) == len(rows), 'Duplicate annotation IDs')
    return result


def request_batches(path):
    batch, count = [], 0
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            require(row['token_count'] == len(row['tokens']), 'Chunk token count mismatch')
            if batch and (len(batch) >= 32 or count + row['token_count'] > 64000):
                yield batch
                batch, count = [], 0
            batch.append(row)
            count += row['token_count']
    if batch:
        yield batch


def verify_corpus(root, source_manifest_sha, expected_posts=60000, expected_dimension=3072):
    """Small fixture sizes are injectable for tests; the CLI fixes production values."""
    root = root.resolve(strict=True)
    transfer_path = safe_path(root, 'transfer_manifest.json')
    require(sha256(transfer_path) == source_manifest_sha, 'Transferred source manifest changed')
    transfer = read_json(transfer_path)
    require(transfer['total_posts'] == expected_posts, 'Unexpected source corpus size')
    require(len(transfer['files']) == transfer['file_count'], 'Transfer file count mismatch')
    prepared = root / PREPARED
    plan = read_json(safe_path(root, PREPARED + '/plan.json'))
    require(plan['embedding_scope'] == 'all_indexed_posts' and
            plan['unique_posts'] == expected_posts, 'Prepared corpus is not the full corpus')
    identity = dict(endpoint=plan['endpoint'], model=MODEL, dimensions=None,
                    encoding=plan['encoding'], max_input_tokens=plan['max_input_tokens'],
                    chunk_overlap_tokens=plan['chunk_overlap_tokens'], pooling=plan['pooling'],
                    chunks_sha256=plan['prepared_sha256']['chunks_restricted.jsonl'],
                    prepared_identity_sha256=digest_json(plan['prepared_sha256']))
    require(identity['endpoint'] == ENDPOINT, 'Unexpected Azure endpoint')
    cache_key = digest_json(identity)
    cache_relative = PREPARED + '/azure_cache_' + cache_key[:16]
    cache = root / cache_relative
    mutable = {PREPARED + '/embedding_status.json', cache_relative + '/request_manifest.json'}
    for relative, expected in transfer['files'].items():
        path = safe_path(root, relative)
        if relative not in mutable:
            require(path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256'],
                    'Original transferred file changed: ' + relative)
    for relative, expected in plan['prepared_sha256'].items():
        require(sha256(safe_path(prepared, relative)) == expected, 'Prepared input changed')

    # Keep the remote writer excluded for the entire completion audit.
    lock_path = safe_path(root, PREPARED + '/.embedding.lock')
    with lock_path.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = verify_outputs(root, prepared, cache, identity, cache_key, plan,
                                expected_posts, expected_dimension)
    result['source_manifest_sha256'] = source_manifest_sha
    return result


def verify_outputs(root, prepared, cache, identity, cache_key, plan, expected_posts, dimension):
    status = read_json(safe_path(prepared, 'embedding_status.json'))
    require(status.get('status') == 'complete', 'Embedding job has not completed')
    require(status['posts'] == expected_posts and status['dimension'] == dimension and
            status['requested_model'] == MODEL and status['response_model'] == MODEL,
            'Completed status has wrong corpus, dimension, or model')
    ledger_path = safe_path(cache, 'request_manifest.json')
    ledger = read_json(ledger_path)
    require(ledger['identity'] == identity and ledger['dimension'] == dimension and
            ledger['response_model'] == MODEL, 'Request cache identity mismatch')
    chunks_seen, batch_names, chunk_ids = 0, set(), {}
    for j, batch in enumerate(request_batches(prepared / 'chunks_restricted.jsonl')):
        name = f'batch_{j:05d}.npz'
        batch_names.add(name)
        path = safe_path(cache, name)
        require(sha256(path) == ledger['completed_batches'].get(name), 'Response batch checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            require(str(data['request_key'].item()) == digest_json({'identity': identity, 'batch': batch}),
                    'Response batch belongs to different posts')
            require(str(data['model'].item()) == MODEL, 'Response batch model mismatch')
            vectors = data['embeddings']
            require(vectors.shape == (len(batch), dimension) and np.isfinite(vectors).all() and
                    np.all(np.linalg.norm(vectors, axis=1) > 0), 'Invalid response batch vectors')
        for row in batch:
            chunk_ids.setdefault(row['annotation_id'], []).append(row['chunk_index'])
        chunks_seen += len(batch)
    require(batch_names == set(ledger['completed_batches']) and
            batch_names == set(ledger['usage']), 'Incomplete or extra request ledger entries')
    require(status['request_batches_cached'] == len(batch_names) and chunks_seen == plan['chunks'],
            'Request count mismatch')

    manifest_path = safe_path(cache, 'embedding_cache_manifest.json')
    manifest = read_json(manifest_path)
    require(manifest['records'] == expected_posts and manifest['cache_key'] == cache_key,
            'Final embedding manifest mismatch')
    require(manifest['azure_request_cache_sha256'] == sha256(ledger_path), 'Final manifest references an old request cache')
    for key, filename in [('builder_sha256', 'build_azure_semantic_pilot.py'),
                          ('client_sha256', 'unc_azure_embeddings.py')]:
        require(manifest[key] == sha256(safe_path(root, 'code/' + filename)), 'Encoder code hash mismatch')
    encoder = manifest['encoder']
    require(encoder['requested_model'] == MODEL and encoder['response_model'] == MODEL and
            encoder['embedding_dimension'] == dimension and encoder['normalize_embeddings'] is True and
            encoder['pooling'] == identity['pooling'] and encoder['endpoint'] == ENDPOINT,
            'Final encoder metadata mismatch')
    posts = read_rows(safe_path(prepared, 'selected_posts_restricted.csv'))
    originals = read_rows(safe_path(prepared, 'post_metadata.csv'))
    index_path = safe_path(cache, 'post_embedding_index.csv')
    index = read_rows(index_path)
    require(len(posts) == expected_posts and set(posts) == set(index) == set(originals) == set(chunk_ids),
            'Final index does not cover every prepared post')
    for aid, post in posts.items():
        require(sorted(chunk_ids[aid]) == list(range(int(post['chunk_count']))), 'Missing or duplicate chunks')
        for field in ('post_id', 'text_sha256'):
            require(index[aid][field] == post[field] == originals[aid][field], 'Post identity mismatch')

    seen = set()
    shards = manifest['completed_shards']
    require(bool(shards), 'No final embedding shards')
    for name, expected in shards.items():
        path = safe_path(cache / 'post_embedding_shards', name)
        require(sha256(path) == expected, 'Final shard checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            ids = data['annotation_ids'].astype(str).tolist()
            vectors = data['embeddings']
            require(str(data['cache_key'].item()) == cache_key, 'Wrong cache key in shard')
            require(vectors.dtype == np.float32 and vectors.shape == (len(ids), dimension) and
                    np.isfinite(vectors).all() and
                    np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=0),
                    'Invalid or unnormalized final vectors')
            require(all(len(data[field]) == len(ids) for field in ('text_sha256', 'token_counts', 'chunk_counts')),
                    'Shard metadata length mismatch')
            for row, aid in enumerate(ids):
                require(aid in posts and aid not in seen, 'Unknown or duplicate post in shards')
                seen.add(aid)
                require(index[aid]['shard'] == name and int(index[aid]['shard_row']) == row,
                        'Index points to the wrong vector')
                require(str(data['text_sha256'][row]) == posts[aid]['text_sha256'], 'Vector text identity mismatch')
                for field, array in [('token_count', 'token_counts'), ('chunk_count', 'chunk_counts')]:
                    require(int(data[array][row]) == int(posts[aid][field]) == int(index[aid][field]),
                            'Vector token or chunk count mismatch')
    require(seen == set(posts), 'Missing posts in finalized shards')
    config_path = safe_path(prepared, 'comparison_config.json')
    require(sha256(config_path) == sha256(safe_path(cache, 'comparison_config.json')), 'Published configuration mismatch')
    config = read_json(config_path)
    require(config['input']['dimension'] == dimension, 'Published dimension mismatch')
    targets = {'index': index_path, 'manifest': manifest_path,
               'roles': prepared / 'roles.csv', 'human_reviews': prepared / 'human_reviews.csv'}
    for key, target in targets.items():
        require(safe_path(root, config['input'][key]) == target and
                config['input_sha256'][key] == sha256(target), 'Published input binding mismatch')
    require(config['input']['shards'] == str((cache / 'post_embedding_shards').relative_to(root)),
            'Published shard path mismatch')
    return {'complete': True, 'posts': len(seen), 'dimension': dimension, 'model': MODEL,
            'request_batches': len(batch_names), 'shards': len(shards),
            'cache_relative': str(cache.relative_to(root)), 'index_sha256': sha256(index_path),
            'embedding_manifest_sha256': sha256(manifest_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-manifest-sha', required=True)
    args = parser.parse_args()
    try:
        result = verify_corpus(args.root, args.source_manifest_sha)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, 'Completion verification failed: ' + str(exc) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
