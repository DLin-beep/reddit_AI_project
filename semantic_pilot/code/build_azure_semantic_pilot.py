#!/usr/bin/env python3
"""Prepare pilot or all indexed posts, cache UNC Azure embeddings, and hand off to the pilot.

prepare performs no embedding requests. embed is the only network-using command;
it requires an explicit deployed model and an environment-held UNC API key.
Original text, labels, historical roles, and GTE artifacts are never modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from run_semantic_region_pilot import ROOT, load_metadata, sha256, write_csv
from unc_azure_embeddings import AzureEmbeddingClient, chunk_tokens, load_tokenizer, pool_post_embeddings


ENDPOINT = 'https://azureaiapi.cloud.unc.edu/openai/v1/embeddings'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(source_config, text_path, output, data_root=ROOT, encoding_name='cl100k_base',
            max_tokens=8191, overlap=128, tokenizer=None, all_posts=False):
    if output.exists():
        raise FileExistsError(f'Prepared directory already exists: {output}')
    config = json.loads(source_config.read_text())
    index, roles, reviews, manifest, selected, paths = load_metadata(config, data_root)
    # Embedding coverage is independent of the downstream pilot fit/evaluation
    # selections, which remain fixed by the original comparison configuration.
    wanted = set(index) if all_posts else set(selected['training']) | set(selected['heldout_historical']) | set(reviews)
    texts = {}
    with text_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            aid = row['annotation_id']
            if aid not in wanted:
                continue
            if aid in texts:
                raise ValueError('Duplicate selected post in text source')
            text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
            if row['post_id'] != index[aid]['post_id'] or text_hash != index[aid]['text_sha256']:
                raise ValueError('Selected post text/identity differs from the authenticated index')
            texts[aid] = row['text']
    if set(texts) != wanted:
        raise ValueError('The text source does not cover every selected post')
    tokenizer = tokenizer or load_tokenizer(encoding_name)
    chunks, posts = [], []
    for aid in sorted(wanted):
        tokens = tokenizer.encode(texts[aid], disallowed_special=())
        split = chunk_tokens(tokens, max_tokens=max_tokens, overlap=overlap)
        posts.append(dict(annotation_id=aid, post_id=index[aid]['post_id'],
                          text_sha256=index[aid]['text_sha256'], token_count=len(tokens), chunk_count=len(split)))
        for j, chunk in enumerate(split):
            chunks.append(dict(annotation_id=aid, chunk_index=j, tokens=chunk,
                               token_count=len(chunk), pool_weight=len(chunk) if j == 0 else len(chunk) - overlap))
    output.mkdir(parents=True)
    write_csv(output / 'post_metadata.csv', list(index.values()))
    write_csv(output / 'roles.csv', list(roles.values()))
    write_csv(output / 'human_reviews.csv', list(reviews.values()))
    write_csv(output / 'selected_posts_restricted.csv', posts)
    with (output / 'chunks_restricted.jsonl').open('w') as stream:
        for row in chunks:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    atomic_json(output / 'comparison_template.json', config)
    files = ['post_metadata.csv', 'roles.csv', 'human_reviews.csv', 'selected_posts_restricted.csv',
             'chunks_restricted.jsonl', 'comparison_template.json']
    plan = dict(status='prepared_no_embedding_requests', prepared_utc=datetime.now(timezone.utc).isoformat(),
                embedding_scope='all_indexed_posts' if all_posts else 'pilot_and_reviewed_posts',
                indexed_posts=len(index),
                endpoint=ENDPOINT, model=None, encoding=encoding_name, max_input_tokens=max_tokens,
                chunk_overlap_tokens=overlap, pooling='normalize_each_chunk_then_new_token_weighted_mean_then_L2',
                training_posts=len(selected['training']), comparison_posts=len(selected['heldout_historical']),
                reviewed_posts=len(reviews), unique_posts=len(posts), chunks=len(chunks),
                planned_input_tokens=sum(row['token_count'] for row in chunks),
                chunked_posts=sum(row['chunk_count'] > 1 for row in posts),
                tokenizer_version=version('tiktoken') if tokenizer.__class__.__module__.startswith('tiktoken') else 'test_tokenizer',
                source_sha256={str(source_config): sha256(source_config), str(text_path): sha256(text_path)},
                prepared_sha256={name: sha256(output / name) for name in files},
                new_human_scores=0, api_requests=0)
    atomic_json(output / 'plan.json', plan)
    return plan


def batches(chunks, max_items=32, max_tokens=64000):
    batch, count = [], 0
    for row in chunks:
        if row['token_count'] > max_tokens:
            raise ValueError('A chunk exceeds the request token budget')
        if batch and (len(batch) >= max_items or count + row['token_count'] > max_tokens):
            yield batch
            batch, count = [], 0
        batch.append(row)
        count += row['token_count']
    if batch:
        yield batch


def verify_prepared(prepared):
    plan = json.loads((prepared / 'plan.json').read_text())
    for name, expected in plan['prepared_sha256'].items():
        if sha256(prepared / name) != expected:
            raise ValueError(f'Prepared input changed: {name}; prepare a new directory')
    return plan


def load_batch(path, request_key, n_rows, expected_hash=None):
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError('Cached Azure response failed its integrity check')
    with np.load(path, allow_pickle=False) as data:
        if str(data['request_key'].item()) != request_key:
            raise ValueError('Cached Azure response belongs to a different request')
        x = data['embeddings'].astype(np.float64)
        model = str(data['model'].item())
        usage = json.loads(str(data['usage_json'].item()))
    if x.ndim != 2 or len(x) != n_rows or x.shape[1] < 2 or not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) == 0) or not model:
        raise ValueError('Invalid cached Azure response')
    return dict(embeddings=x, model=model, usage=usage)


def relative_path(path, data_root):
    return os.path.relpath(path.resolve(), data_root.resolve())


def finalize(prepared, cache_dir, identity, ledger, chunks, responses, data_root):
    with (prepared / 'post_metadata.csv').open(newline='') as stream:
        metadata = list(csv.DictReader(stream))
    with (prepared / 'selected_posts_restricted.csv').open(newline='') as stream:
        posts = {r['annotation_id']: r for r in csv.DictReader(stream)}
    grouped = defaultdict(list)
    for row, vector in zip(chunks, responses):
        grouped[row['annotation_id']].append((row['chunk_index'], vector, row['pool_weight']))
    if set(grouped) != set(posts) or len(chunks) != len(responses):
        raise ValueError('Cannot finalize an incomplete embedding cache')
    vectors = {}
    for aid, parts in grouped.items():
        parts.sort(key=lambda part: part[0])
        vectors[aid] = pool_post_embeddings(np.asarray([p[1] for p in parts]), np.asarray([p[2] for p in parts]))
    dimension = ledger['dimension']
    shard_dir = cache_dir / 'post_embedding_shards'
    shard_dir.mkdir(exist_ok=True)
    manifest = {'schema_version': 1, 'cache_key': digest_json(identity), 'records': len(vectors),
                'builder_sha256': sha256(Path(__file__)),
                'client_sha256': sha256(Path(__file__).with_name('unc_azure_embeddings.py')),
                'encoder': {'provider': 'UNC_Azure_OpenAI', 'endpoint': identity['endpoint'],
                            'requested_model': identity['model'], 'response_model': ledger['response_model'],
                            'requested_dimensions': identity['dimensions'], 'embedding_dimension': dimension,
                            'encoding': identity['encoding'], 'normalize_embeddings': True,
                            'pooling': identity['pooling'], 'model_version': 'not separately exposed by this endpoint'},
                'completed_shards': {}, 'azure_request_cache_sha256': sha256(cache_dir / 'request_manifest.json')}
    locations = {}
    ids = sorted(vectors)
    for start in range(0, len(ids), 256):
        part = ids[start:start + 256]
        name = f'posts_{start:06d}_{start + len(part):06d}.npz'
        target = shard_dir / name
        matrix = np.asarray([vectors[i] for i in part], dtype=np.float32)
        if target.exists():
            with np.load(target, allow_pickle=False) as old:
                if (str(old['cache_key'].item()) != manifest['cache_key'] or
                    not np.array_equal(old['annotation_ids'].astype(str), part) or
                    not np.array_equal(old['text_sha256'].astype(str), [posts[i]['text_sha256'] for i in part]) or
                    not np.array_equal(old['embeddings'], matrix)):
                    raise ValueError('Existing finalized shard differs; refusing to replace it')
        else:
            temporary = shard_dir / (name + '.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, cache_key=np.asarray(manifest['cache_key']), annotation_ids=np.asarray(part),
                         text_sha256=np.asarray([posts[i]['text_sha256'] for i in part]), embeddings=matrix,
                         token_counts=np.asarray([int(posts[i]['token_count']) for i in part]),
                         chunk_counts=np.asarray([int(posts[i]['chunk_count']) for i in part]))
            temporary.replace(target)
        manifest['completed_shards'][name] = sha256(target)
        for j, aid in enumerate(part):
            locations[aid] = (name, j)
    for row in metadata:
        aid = row['annotation_id']
        row['shard'], row['shard_row'] = locations.get(aid, ('NOT_EMBEDDED_IN_AZURE_PILOT', ''))
        row['token_count'] = posts[aid]['token_count'] if aid in posts else ''
        row['chunk_count'] = posts[aid]['chunk_count'] if aid in posts else ''
    write_csv(cache_dir / 'post_embedding_index.csv', metadata)
    atomic_json(cache_dir / 'embedding_cache_manifest.json', manifest)
    config = json.loads((prepared / 'comparison_template.json').read_text())
    sources = dict(index=cache_dir / 'post_embedding_index.csv', manifest=cache_dir / 'embedding_cache_manifest.json',
                   roles=prepared / 'roles.csv', human_reviews=prepared / 'human_reviews.csv')
    for key, path in sources.items():
        config['input'][key] = relative_path(path, data_root)
    config['input']['shards'] = relative_path(shard_dir, data_root)
    config['input']['dimension'] = dimension
    config['input_sha256'] = {key: sha256(path) for key, path in sources.items()}
    # Preserve the declared concentration-per-dimension ceiling when the Azure
    # model uses a different vector dimension from the original GTE template.
    template_dimension = json.loads((prepared / 'comparison_template.json').read_text())['input']['dimension']
    config['models']['vmf_kappa_max'] *= dimension / template_dimension
    config['name'] = 'small_semantic_region_comparison_azure'
    config['azure_embedding_identity'] = copy.deepcopy(identity)
    config['azure_embedding_identity']['response_model'] = ledger['response_model']
    atomic_json(cache_dir / 'comparison_config.json', config)
    atomic_json(prepared / 'comparison_config.json', config)
    summary = {'status': 'complete', 'provider': 'UNC_Azure_OpenAI', 'requested_model': identity['model'],
               'response_model': ledger['response_model'], 'dimension': dimension, 'posts': len(posts),
               'successful_response_prompt_tokens': sum(v.get('prompt_tokens', 0) for v in ledger['usage'].values()),
               'usage_note': 'Successful responses only; failed or uncertain requests can also consume service quota.',
               'request_batches_cached': len(ledger['completed_batches']), 'comparison_config': relative_path(prepared / 'comparison_config.json', data_root)}
    atomic_json(prepared / 'embedding_status.json', summary)
    return summary


def embed(prepared, model, data_root=ROOT, dimensions=None, client=None, max_total_tokens=None,
          request_timeout=60):
    if (isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float))
            or not math.isfinite(request_timeout) or request_timeout <= 0):
        raise ValueError('Request timeout must be a finite positive number of seconds')
    placeholders = ('string', 'your_model_name', 'model_name',
                    'your_large_model_deployment', 'your_embedding_deployment')
    if not isinstance(model, str) or not model.strip() or model.strip().lower() in placeholders:
        raise ValueError('UNC_EMBEDDING_MODEL is missing or still a placeholder. '
                         'Set the exact UNC deployment name for text-embedding-3-large from your API code or portal.')
    plan = verify_prepared(prepared)
    if max_total_tokens is not None and plan['planned_input_tokens'] > max_total_tokens:
        raise ValueError('Planned input tokens exceed the supplied token limit')
    identity = dict(endpoint=plan['endpoint'], model=model.strip(), dimensions=dimensions,
                    encoding=plan['encoding'], max_input_tokens=plan['max_input_tokens'],
                    chunk_overlap_tokens=plan['chunk_overlap_tokens'], pooling=plan['pooling'],
                    chunks_sha256=plan['prepared_sha256']['chunks_restricted.jsonl'],
                    prepared_identity_sha256=digest_json(plan['prepared_sha256']))
    cache_dir = prepared / ('azure_cache_' + digest_json(identity)[:16])
    cache_dir.mkdir(exist_ok=True)
    # Different model caches still publish one prepared-directory handoff, so
    # serialize them together rather than locking only each individual cache.
    with (prepared / '.embedding.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = cache_dir / 'request_manifest.json'
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            if ledger['identity'] != identity:
                raise ValueError('Existing request cache has a different identity')
        else:
            ledger = dict(identity=identity, dimension=None, response_model=None, completed_batches={}, usage={})
        chunks = read_jsonl(prepared / 'chunks_restricted.jsonl')
        responses = []
        requests = list(batches(chunks))
        atomic_json(prepared / 'embedding_status.json', {'status': 'running', 'model': model, 'cached_batches': len(ledger['completed_batches']), 'total_batches': len(requests)})
        try:
            for j, batch in enumerate(requests):
                name = f'batch_{j:05d}.npz'
                path = cache_dir / name
                key = digest_json({'identity': identity, 'batch': batch})
                expected = ledger['completed_batches'].get(name)
                if expected and not path.exists():
                    raise ValueError('A committed response cache file is missing')
                if path.exists():
                    reply = load_batch(path, key, len(batch), expected_hash=expected)
                else:
                    if client is None:
                        api_key = os.environ.get('UNC_AI_API_KEY', '').strip()
                        if not api_key:
                            raise ValueError('Set UNC_AI_API_KEY in the execution environment')
                        if api_key in ('123', 'YOUR_API_KEY'):
                            raise ValueError('UNC_AI_API_KEY is still an example key. Set your real UNC API key locally.')
                        client = AzureEmbeddingClient(plan['endpoint'], model, api_key,
                                                      dimensions=dimensions, timeout=request_timeout)
                    reply = client.embed([row['tokens'] for row in batch])
                    temporary = cache_dir / (name + '.tmp')
                    with temporary.open('wb') as stream:
                        np.savez(stream, request_key=np.asarray(key), embeddings=reply['embeddings'],
                                 model=np.asarray(reply['model']), usage_json=np.asarray(json.dumps(reply['usage'])))
                    temporary.replace(path)
                    reply = load_batch(path, key, len(batch))
                dimension = reply['embeddings'].shape[1]
                if dimensions is not None and dimension != dimensions:
                    raise ValueError('Response dimension differs from requested dimensions')
                if ledger['dimension'] not in (None, dimension) or ledger['response_model'] not in (None, reply['model']):
                    raise ValueError('Azure response model or dimension changed during this cache; refusing to mix embeddings')
                ledger['dimension'], ledger['response_model'] = dimension, reply['model']
                ledger['completed_batches'][name] = sha256(path)
                ledger['usage'][name] = reply['usage']
                atomic_json(ledger_path, ledger)
                responses.extend(reply['embeddings'])
                print(f'Azure batch {j + 1}/{len(requests)} cached', flush=True)
            return finalize(prepared, cache_dir, identity, ledger, chunks, responses, data_root)
        except BaseException as exc:
            # Exception strings from transport are sanitized by the client. Do
            # not persist server bodies, request headers, keys, or input text.
            atomic_json(prepared / 'embedding_status.json', {'status': 'incomplete', 'error_type': type(exc).__name__,
                        'error_category': getattr(exc, 'category', None),
                        'request_attempts': getattr(exc, 'attempts', None),
                        'cached_batches': len(ledger['completed_batches']), 'total_batches': len(requests)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Select and tokenize posts without embedding requests')
    prep.add_argument('--all-posts', action='store_true',
                      help='Embed every indexed post; keep the downstream pilot fit/evaluation selections unchanged')
    prep.add_argument('--source-config', type=Path, default=ROOT / 'semantic_region_pilot_config.json')
    prep.add_argument('--texts', type=Path, default=ROOT / 'output/stm_pilot_20260908/input/05_annotation_prep/annotation_posts_blinded.csv')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--data-root', type=Path, default=ROOT)
    prep.add_argument('--encoding', default='cl100k_base')
    prep.add_argument('--max-input-tokens', type=int, default=8191)
    prep.add_argument('--chunk-overlap', type=int, default=128)
    api = sub.add_parser('embed', help='Generate or resume embeddings through the UNC endpoint')
    api.add_argument('--prepared', type=Path, required=True)
    api.add_argument('--model', required=True)
    api.add_argument('--dimensions', type=int)
    api.add_argument('--max-total-tokens', type=int)
    api.add_argument('--request-timeout', type=float, default=60,
                     help='Timeout in seconds for each API attempt; changing it preserves cached vectors')
    api.add_argument('--data-root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.source_config, args.texts, args.output, args.data_root,
                             args.encoding, args.max_input_tokens, args.chunk_overlap,
                             all_posts=args.all_posts)
            result = {k: result[k] for k in ('status', 'embedding_scope', 'indexed_posts',
                                           'training_posts', 'comparison_posts', 'reviewed_posts',
                                           'unique_posts', 'chunks', 'planned_input_tokens', 'chunked_posts', 'api_requests')}
        else:
            result = embed(args.prepared, args.model, args.data_root, args.dimensions,
                           max_total_tokens=args.max_total_tokens, request_timeout=args.request_timeout)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f'Azure preparation/generation stopped: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
