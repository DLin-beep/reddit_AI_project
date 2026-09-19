#!/usr/bin/env python3
"""Fit independent semantic maps on balanced reference folds, with resumable tasks.

This is a measurement pilot: event effects and final map selection are out of
scope. Historical STM roles are retained as metadata, not used for the new split.
No input file is modified and no embedding API is called.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_semantic_region_pilot import load_vectors, predict as predict_directional, read_csv, sha256, unique_rows
from semantic_threefold_density import fit_density_regions, predict_density_regions
from semantic_region_directional import fit_spherical_kmeans, fit_vmf

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ('A', 'B', 'C')
TRAIN_SETS = (('A',), ('B',), ('C',), ('A', 'B'), ('A', 'C'), ('B', 'C'))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def positive_integer(value, name, upper=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (upper is not None and value > upper):
        raise ValueError(f'{name} must be a positive integer' + (f' <= {upper}' if upper else ''))
    return value


def validate_config(config):
    dimension = positive_integer(config['input']['dimension'], 'dimension', 65536)
    if dimension < 2:
        raise ValueError('Embedding dimension must be at least two')
    split = config.get('sampling', {})
    communities = positive_integer(split.get('expected_communities', 200), 'expected_communities')
    per = positive_integer(split.get('expected_posts_per_community', 300), 'expected_posts_per_community')
    count = positive_integer(split.get('expected_posts', 60000), 'expected_posts')
    if per % 3 or count != communities * per:
        raise ValueError('Expected balanced corpus must divide exactly into three equal community-balanced folds')
    if isinstance(split.get('seed', 20260914), bool) or not isinstance(split.get('seed', 20260914), int):
        raise ValueError('Split seed must be an integer')
    resources = config['resources']
    positive_integer(resources['threads'], 'threads', 64)
    limit = resources['fit_wall_limit_seconds']
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not np.isfinite(limit) or not 0 < limit <= 86400:
        raise ValueError('Fit wall limit must be finite and between zero and 86400 seconds')
    settings = config['models']
    positive_integer(settings['max_iter'], 'max_iter', 1000)
    for name in ('tolerance', 'vmf_kappa_max'):
        if not isinstance(settings[name], (int, float)) or not np.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    positive_integer(settings['density_min_region_size'], 'density_min_region_size')
    families = settings.get('families', ['density', 'vmf', 'spherical_kmeans'])
    if not families or len(families) != len(set(families)) or set(families) - {'density', 'vmf', 'spherical_kmeans'}:
        raise ValueError('Unsupported or duplicate model families')
    for name in ('k', 'seeds', 'density_neighbors', 'density_quantiles'):
        values = settings[name]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f'{name} must be a nonempty list without duplicates')
        for value in values:
            if name == 'seeds':
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**32 - 1:
                    raise ValueError('Seeds must be unsigned 32-bit integers')
            elif name == 'density_quantiles':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Density quantiles must be in [0, 1]')
            else:
                positive_integer(value, name)
                if value >= count // 3:
                    raise ValueError(f'{name} must be smaller than a single training fold')
    if len(planned_tasks(config)) > 72:
        raise ValueError('This bounded pilot permits at most 72 fit tasks; narrow the grid')


def planned_tasks(config):
    settings = config['models']
    families = settings.get('families', ['density', 'vmf', 'spherical_kmeans'])
    specs = []
    for family in ('spherical_kmeans', 'vmf'):
        if family in families:
            for k in settings['k']:
                for seed in settings['seeds']:
                    specs.append({'family': family, 'base': f'{family}_k{k}', 'k': k, 'seed': seed})
    if 'density' in families:
        for neighbors in settings['density_neighbors']:
            for q in settings['density_quantiles']:
                specs.append({'family': 'density', 'base': f'density_nn{neighbors}_q{q:g}', 'neighbors': neighbors, 'quantile': q, 'seed': None})
    tasks = []
    for train_folds in TRAIN_SETS:
        for spec in specs:
            variant = '' if spec['seed'] is None else f"_seed{spec['seed']}"
            tasks.append(dict(spec, task_id=len(tasks), fit_id=f"{spec['base']}{variant}_train{''.join(train_folds)}",
                              train_folds=list(train_folds), eval_folds=[fold for fold in FOLDS if fold not in train_folds]))
    return tasks


def make_folds(index, sampling):
    by_community = defaultdict(list)
    for aid, row in index.items():
        by_community[row['subreddit']].append(aid)
    expected_communities = sampling.get('expected_communities', 200)
    expected_per = sampling.get('expected_posts_per_community', 300)
    expected_posts = sampling.get('expected_posts', 60000)
    if len(index) != expected_posts or len(by_community) != expected_communities:
        raise ValueError('Reference corpus count or community count differs from the declared design')
    if any(len(ids) != expected_per for ids in by_community.values()):
        raise ValueError('Every community must have the declared number of reference posts')
    if expected_per % 3:
        raise ValueError('Posts per community must be divisible by three')
    seed = sampling.get('seed', 20260914)
    folds = {fold: [] for fold in FOLDS}
    for community, ids in sorted(by_community.items()):
        ordered = sorted(ids, key=lambda aid: (hashlib.sha256(f'{seed}|{aid}'.encode()).hexdigest(), aid))
        for position, aid in enumerate(ordered):
            folds[FOLDS[position % 3]].append(aid)
    return folds


def load_inputs(config, data_root):
    validate_config(config)
    data_root = Path(data_root)
    paths = {key: data_root / config['input'][key] for key in ('index', 'manifest', 'roles', 'human_reviews')}
    if config['input'].get('texts'):
        paths['texts'] = data_root / config['input']['texts']
    hashes = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {key} input: {path}')
        hashes[key] = sha256(path)
        expected = config.get('input_sha256', {}).get(key)
        if expected is None:
            raise ValueError(f'An authenticated input_sha256 is required for {key}')
        if hashes[key] != expected:
            raise ValueError(f'Frozen input hash mismatch: {key}')
    index = unique_rows(read_csv(paths['index']))
    if len({row.get('post_id') for row in index.values()}) != len(index):
        raise ValueError('Duplicate post_id across annotation IDs would leak physical posts between folds')
    roles = unique_rows(read_csv(paths['roles']))
    reviews = unique_rows(read_csv(paths['human_reviews']))
    if set(index) != set(roles):
        raise ValueError('Historical roles must cover exactly the indexed posts')
    for aid, row in index.items():
        for key in ('post_id', 'text_sha256', 'subreddit', 'year_month', 'shard', 'shard_row'):
            if not row.get(key):
                raise ValueError(f'Embedding index has an empty {key}')
        if roles[aid].get('sample_role') not in ('training', 'heldout_historical'):
            raise ValueError('Unrecognized historical role')
        for key in ('post_id', 'text_sha256'):
            if roles[aid].get(key) and roles[aid][key] != row[key]:
                raise ValueError(f'Historical role identity mismatch: {key}')
    for aid, row in reviews.items():
        if aid not in index or any(row.get(key) != index[aid][key] for key in ('post_id', 'text_sha256')):
            raise ValueError('Human-review identity/text hash mismatch')
        for rater in ('derek', 'arya'):
            if row.get(f'{rater}_personal_req_score') not in ('0', '1', '2', '3'):
                raise ValueError('Missing or invalid existing human score')
    expected_reviews = config.get('sampling', {}).get('expected_human_reviews', 300)
    if len(reviews) != expected_reviews:
        raise ValueError('Human-review count differs from the declared design')
    manifest = json.loads(paths['manifest'].read_text())
    dimension = config['input']['dimension']
    if manifest.get('records') != len(index) or manifest.get('encoder', {}).get('embedding_dimension') != dimension:
        raise ValueError('Embedding manifest record count or dimension mismatch')
    if not manifest.get('encoder', {}).get('normalize_embeddings'):
        raise ValueError('This pilot requires normalized embeddings')
    if not isinstance(manifest.get('cache_key'), str) or not manifest['cache_key']:
        raise ValueError('Embedding cache identity is missing')
    folds = make_folds(index, config.get('sampling', {}))
    shard_dir = data_root / config['input']['shards']
    names = {row['shard'] for row in index.values()}
    for name in sorted(names):
        if Path(name).name != name or name not in manifest.get('completed_shards', {}):
            raise ValueError('Unrecognized embedding shard in index')
        if not (shard_dir / name).is_file():
            raise FileNotFoundError(f'Required embedding shard is missing: {name}')
    return dict(index=index, roles=roles, reviews=reviews, manifest=manifest, folds=folds,
                paths=paths, input_sha256=hashes, shard_dir=shard_dir)


def fold_rows(inputs):
    return [dict(annotation_id=aid, post_id=inputs['index'][aid]['post_id'], subreddit=inputs['index'][aid]['subreddit'],
                 year_month=inputs['index'][aid]['year_month'], text_sha256=inputs['index'][aid]['text_sha256'],
                 historical_role=inputs['roles'][aid]['sample_role'], fold=fold, existing_human_review=int(aid in inputs['reviews']))
            for fold in FOLDS for aid in inputs['folds'][fold]]


def input_summary(config, inputs):
    sources = [Path(__file__), Path(__file__).with_name('run_semantic_region_pilot.py'),
               Path(__file__).with_name('semantic_region_directional.py'), Path(__file__).with_name('semantic_region_density.py'),
               Path(__file__).with_name('semantic_threefold_density.py')]
    report = Path(__file__).with_name('semantic_threefold_report.py')
    if report.is_file():
        sources.append(report)
    summary = {'posts': len(inputs['index']), 'communities': len({row['subreddit'] for row in inputs['index'].values()}),
               'fold_posts': {fold: len(ids) for fold, ids in inputs['folds'].items()},
               'human_review_posts': len(inputs['reviews']), 'dimension': config['input']['dimension'],
               'input_sha256': inputs['input_sha256'], 'embedding_cache_key': inputs['manifest']['cache_key'],
               'fold_sha256': digest(fold_rows(inputs)), 'config_sha256': digest(config),
               'code_sha256': {path.name: sha256(path) for path in sources},
               'runtime': {'python': sys.version, **{name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')}},
               'historical_roles': 'Preserved as metadata only; this resplit is not a historically untouched validation set.',
               'evaluation_scope': 'Reference-data measurement reproducibility; no event responses or final map selection.'}
    summary['input_fingerprint'] = digest(summary)
    return summary


@contextmanager
def file_lock(path, blocking=False):
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f'Another process holds the lock: {path}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def initialize_run(config, inputs, output):
    output = Path(output)
    if output.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    output.mkdir(parents=True, exist_ok=True)
    summary = input_summary(config, inputs)
    with file_lock(output / '.run.lock', blocking=True):
        existing = output / 'inputs.json'
        if existing.exists():
            previous = json.loads(existing.read_text())
            if previous != summary:
                raise ValueError('Existing run has different configuration, inputs, split, or source code; choose a new output directory')
            if json.loads((output / 'config.json').read_text()) != config or json.loads((output / 'tasks.json').read_text()) != planned_tasks(config):
                raise ValueError('Existing run metadata was modified')
            if read_csv(output / 'fold_assignments_restricted.csv') != [{key: str(value) for key, value in row.items()} for row in fold_rows(inputs)]:
                raise ValueError('Existing run fold assignments were modified')
        else:
            unexpected = set(path.name for path in output.iterdir()) - {'.run.lock'}
            if unexpected:
                raise FileExistsError('Refusing to initialize a nonempty directory without authenticated run metadata')
            (output / 'tasks').mkdir()
            (output / 'locks').mkdir()
            (output / 'incomplete').mkdir()
            write_json(output / 'config.json', config)
            write_json(output / 'tasks.json', planned_tasks(config))
            write_csv(output / 'fold_assignments_restricted.csv', fold_rows(inputs))
            write_json(output / 'inputs.json', summary)
            write_json(output / 'status.json', {'status': 'prepared', 'planned_fits': len(planned_tasks(config)), 'completed_fits': 0,
                                               'input_fingerprint': summary['input_fingerprint'], 'prepared_utc': now()})
    return summary


def validate_completed_task(task_dir, spec, summary):
    status_path = task_dir / 'status.json'
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text())
    if status.get('input_fingerprint') != summary['input_fingerprint'] or status.get('task_id') != spec['task_id']:
        raise ValueError('Existing task belongs to a different input/configuration or task')
    if status.get('status') != 'complete':
        return False
    hashes = status.get('artifact_sha256', {})
    required = {'fit.json', 'model.npz', 'predictions.npz', 'review_assignments_restricted.csv'}
    if set(hashes) != required:
        raise ValueError('Completed task artifact inventory is invalid')
    for name, expected in hashes.items():
        path = task_dir / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Completed task artifact is missing or modified: {name}')
    if json.loads((task_dir / 'fit.json').read_text()).get('task') != spec:
        raise ValueError('Completed task settings differ from the plan')
    return True


def centrality(X, labels):
    result = np.full(len(X), np.nan, dtype=np.float32)
    for region in np.unique(labels[labels >= 0]):
        members = labels == region
        centroid = X[members].sum(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            result[members] = X[members] @ (centroid / norm)
    return result


def scalar_model_metadata(model):
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in model.items()
            if isinstance(value, (str, int, float, bool, np.generic)) and not isinstance(value, np.ndarray)}


def predict(family, X, model, threads):
    if family == 'density':
        return predict_density_regions(X, model, threads=threads), None, None, None
    return predict_directional(family, X, model)


def evaluation_summary(spec, fold, model, labels, confidence, log_density, settings, training_posts):
    accepted = labels >= 0
    counts = Counter(labels[accepted].tolist())
    row = {'fit': spec['fit_id'], 'fit_id': spec['fit_id'], 'task_id': spec['task_id'], 'family': spec['family'], 'base': spec['base'],
           'training_folds': ''.join(spec['train_folds']), 'evaluation_fold': fold, 'training_posts': training_posts,
           'training_regions': len(np.unique(model['labels'][model['labels'] >= 0])),
           'training_coverage': float(np.mean(model['labels'] >= 0)), 'evaluation_posts': len(labels),
           'evaluation_regions': len(counts), 'evaluation_coverage': float(accepted.mean()),
           'largest_region_share_all_evaluation': max(counts.values(), default=0) / len(labels),
           'iterations': model.get('iterations'), 'converged': model.get('converged', 'deterministic')}
    if spec['family'] == 'vmf':
        row.update(mean_evaluation_log_density=float(log_density.mean()),
                   concentration_cap_hits=int(np.sum(np.isclose(model['kappas'], settings['vmf_kappa_max']))),
                   mean_max_membership=float(confidence.mean()))
    elif spec['family'] == 'spherical_kmeans':
        row['mean_evaluation_cosine_distance'] = float(np.mean(1 - confidence))
    else:
        row['threshold_radius'] = float(model['threshold_radius'])
    return row


def execute_fit(config, inputs, spec, task_dir, summary):
    """Run one fit; callers own the task lock and time budget."""
    ids = [aid for fold in FOLDS for aid in inputs['folds'][fold]]
    vectors, verified = load_vectors(inputs['index'], ids, inputs['manifest'], inputs['shard_dir'], config['input']['dimension'])
    train_ids = [aid for fold in spec['train_folds'] for aid in inputs['folds'][fold]]
    X = np.asarray([vectors[aid] for aid in train_ids], dtype=np.float64)
    settings = config['models']
    started = time.monotonic()
    if spec['family'] == 'spherical_kmeans':
        model = fit_spherical_kmeans(X, spec['k'], spec['seed'], max_iter=settings['max_iter'], tol=settings['tolerance'])
    elif spec['family'] == 'vmf':
        model = fit_vmf(X, spec['k'], spec['seed'], kappa_max=settings['vmf_kappa_max'], max_iter=settings['max_iter'], tol=settings['tolerance'])
    else:
        model = fit_density_regions(X, spec['neighbors'], spec['quantile'], settings['density_min_region_size'],
                                    threads=config['resources']['threads'], neighbor_cache=task_dir.parents[1] / 'neighbor_cache',
                                    cache_key=digest({'training_ids': train_ids, 'encoder': inputs['manifest']['cache_key']}),
                                    max_neighbors=max(settings['density_neighbors']))
    fit_seconds = time.monotonic() - started
    arrays = {'training_ids': np.asarray(train_ids), 'training_labels': np.asarray(model['labels'], dtype=np.int32)}
    predictions_by_id = {}
    summaries = []
    for fold in spec['eval_folds']:
        eval_ids = inputs['folds'][fold]
        if set(train_ids) & set(eval_ids):
            raise ValueError('Training/evaluation overlap detected')
        E = np.asarray([vectors[aid] for aid in eval_ids], dtype=np.float64)
        labels, confidence, probabilities, log_density = predict(spec['family'], E, model, config['resources']['threads'])
        arrays[f'evaluation_ids_{fold}'] = np.asarray(eval_ids)
        arrays[f'evaluation_labels_{fold}'] = labels.astype(np.int32)
        arrays[f'evaluation_centrality_{fold}'] = centrality(E, labels)
        if confidence is not None:
            arrays[f'evaluation_confidence_{fold}'] = confidence.astype(np.float32)
        if probabilities is not None:
            arrays[f'evaluation_memberships_{fold}'] = probabilities.astype(np.float32)
        if log_density is not None:
            arrays[f'evaluation_log_density_{fold}'] = log_density
        predictions_by_id.update(zip(eval_ids, labels.tolist()))
        row = evaluation_summary(spec, fold, model, labels, confidence, log_density, settings, len(X))
        row['seconds'] = fit_seconds
        summaries.append(row)
        del E
    # Reuse predictions; training reviews use actual fitted labels, especially
    # for density, where the out-of-sample extension is a different rule.
    training_labels = dict(zip(train_ids, model['labels'].tolist()))
    all_labels = dict(predictions_by_id)
    all_labels.update(training_labels)
    review_ids = sorted(inputs['reviews'])
    fold_by_id = {aid: fold for fold in FOLDS for aid in inputs['folds'][fold]}
    arrays['review_ids'] = np.asarray(review_ids)
    arrays['review_labels'] = np.asarray([all_labels[aid] for aid in review_ids], dtype=np.int32)
    arrays['review_used_in_fit'] = np.asarray([aid in training_labels for aid in review_ids], dtype=bool)
    review_rows = [dict(inputs['reviews'][aid], fit=spec['fit_id'], fit_id=spec['fit_id'], annotation_id=aid,
                        historical_role=inputs['roles'][aid]['sample_role'], used_in_fit=int(aid in training_labels),
                        fold=fold_by_id[aid], region=int(all_labels[aid]),
                        assignment_basis='fitted_training_label' if aid in training_labels else 'out_of_training_prediction') for aid in review_ids]
    for row in summaries:
        row['human_review_coverage'] = float(np.mean(arrays['review_labels'] >= 0)) if review_ids else None
    model_arrays = {key: value for key, value in model.items() if isinstance(value, np.ndarray) and key != 'train_X'}
    model_arrays['training_ids'] = np.asarray(train_ids)
    model_arrays['input_fingerprint'] = np.asarray(summary['input_fingerprint'])
    np.savez_compressed(task_dir / 'model.npz', **model_arrays)
    np.savez_compressed(task_dir / 'predictions.npz', **arrays)
    write_csv(task_dir / 'review_assignments_restricted.csv', review_rows,
              fields=None if review_rows else ['fit_id', 'annotation_id', 'region', 'used_in_fit', 'fold'])
    write_json(task_dir / 'fit.json', {'task': spec, 'summaries': summaries, 'model_scalars': scalar_model_metadata(model),
                                     'fit_seconds': fit_seconds, 'input_fingerprint': summary['input_fingerprint'],
                                     'config_sha256': summary['config_sha256'], 'verified_shard_sha256': verified,
                                     'density_training_vectors': 'Reconstruct by training_ids from authenticated canonical shards; not duplicated here.',
                                     'training_embedding_dimension': X.shape[1]})


def run_task(config, inputs, output, task_id, summary=None):
    summary = summary or initialize_run(config, inputs, output)
    tasks = planned_tasks(config)
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < len(tasks):
        raise ValueError('Task ID is outside the planned grid')
    spec = tasks[task_id]
    output = Path(output)
    task_dir = output / 'tasks' / f'task_{task_id:03d}'
    with file_lock(output / 'locks' / f'task_{task_id:03d}.lock'):
        if task_dir.exists():
            if validate_completed_task(task_dir, spec, summary):
                print(json.dumps({'status': 'already_complete', 'task_id': task_id, 'fit_id': spec['fit_id']}), flush=True)
                return 0
            # Failed/interrupted attempt artifacts remain available for audit.
            attempt = 1
            while (output / 'incomplete' / f'task_{task_id:03d}_attempt_{attempt:03d}').exists():
                attempt += 1
            task_dir.rename(output / 'incomplete' / f'task_{task_id:03d}_attempt_{attempt:03d}')
        task_dir.mkdir()
        state = {'status': 'running', 'task_id': task_id, 'fit_id': spec['fit_id'], 'started_utc': now(),
                 'input_fingerprint': summary['input_fingerprint'], 'config_sha256': summary['config_sha256']}
        write_json(task_dir / 'status.json', state)
        started = time.monotonic()
        previous = signal.getsignal(signal.SIGALRM)
        def timed_out(signum, frame):
            raise TimeoutError('Task wall-time budget reached')
        signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, config['resources']['fit_wall_limit_seconds'])
        try:
            with threadpool_limits(limits=config['resources']['threads']):
                execute_fit(config, inputs, spec, task_dir, summary)
            state.update(status='complete', completed_utc=now(), elapsed_seconds=time.monotonic() - started,
                         artifact_sha256={name: sha256(task_dir / name) for name in
                                          ('fit.json', 'model.npz', 'predictions.npz', 'review_assignments_restricted.csv')})
            write_json(task_dir / 'status.json', state)
            print(json.dumps({'status': 'complete', 'task_id': task_id, 'fit_id': spec['fit_id'], 'seconds': state['elapsed_seconds']}), flush=True)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'timeout' if isinstance(exc, TimeoutError) else 'failed',
                         error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic() - started, stopped_utc=now())
            write_json(task_dir / 'status.json', state)
            print(json.dumps({'status': state['status'], 'task_id': task_id, 'error_type': type(exc).__name__, 'error': str(exc)}), file=sys.stderr, flush=True)
            return 130 if isinstance(exc, KeyboardInterrupt) else 2
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def load_texts(inputs):
    """Pass authenticated restricted source records to reporting; no new labels."""
    if 'texts' not in inputs['paths']:
        return {}
    path = inputs['paths']['texts']
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = read_csv(path)
    rows = unique_rows(rows)
    for aid, row in rows.items():
        if aid not in inputs['index']:
            raise ValueError('Text source contains a post outside the authenticated corpus')
        for key in ('post_id', 'text_sha256'):
            if row.get(key) and row[key] != inputs['index'][aid][key]:
                raise ValueError(f'Text source identity mismatch: {key}')
    return rows


def aggregate_run(config, inputs, output, summary):
    from semantic_threefold_report import report_run
    # Authentication failures are fatal, while missing/failed tasks are reported
    # as incomplete evidence by the report builder.
    for spec in planned_tasks(config):
        task_dir = Path(output) / 'tasks' / f"task_{spec['task_id']:03d}"
        if task_dir.exists():
            validate_completed_task(task_dir, spec, summary)
    with file_lock(Path(output) / '.aggregate.lock'):
        report = report_run(output, inputs['index'], inputs['reviews'], texts=load_texts(inputs))
        status_path = Path(output) / 'status.json'
        state = json.loads(status_path.read_text())
        state.update(report, input_fingerprint=summary['input_fingerprint'], aggregated_utc=now())
        write_json(status_path, state)
    print(json.dumps(report, default=str, allow_nan=False), flush=True)
    return 0 if report.get('status') == 'complete' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true', help='Authenticate metadata and all 60k vectors without fitting')
    parser.add_argument('--task-id', type=int, help='Run or resume one zero-based array task')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate complete task artifacts, explicitly reporting missing fits')
    args = parser.parse_args(argv)
    if sum((args.preflight, args.task_id is not None, args.aggregate)) > 1:
        parser.error('--preflight, --task-id and --aggregate are mutually exclusive')
    if not args.preflight and args.output is None:
        parser.error('--output is required for fitting or aggregation')
    try:
        config = json.loads(args.config.read_text())
        inputs = load_inputs(config, args.data_root)
        if args.preflight:
            ids = [aid for fold in FOLDS for aid in inputs['folds'][fold]]
            vectors, hashes = load_vectors(inputs['index'], ids, inputs['manifest'], inputs['shard_dir'], config['input']['dimension'])
            del vectors
            summary = input_summary(config, inputs)
            result = dict(summary, status='ready', planned_fits=len(planned_tasks(config)), authenticated_shards=len(hashes))
            if args.output:
                initialize_run(config, inputs, args.output)
                write_json(args.output / 'preflight.json', dict(result, verified_shard_sha256=hashes, checked_utc=now()))
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        summary = initialize_run(config, inputs, args.output)
        if args.aggregate:
            return aggregate_run(config, inputs, args.output, summary)
        if args.task_id is not None:
            return run_task(config, inputs, args.output, args.task_id, summary)
        failures = 0
        for spec in planned_tasks(config):
            result = run_task(config, inputs, args.output, spec['task_id'], summary)
            failures += int(result != 0)
            if result == 130:
                return result
        aggregate_run(config, inputs, args.output, summary)
        return 2 if failures else 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        print(f'Cannot run threefold pilot: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
