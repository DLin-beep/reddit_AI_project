#!/usr/bin/env python3
"""Run the bounded post-embedding comparison; never fetch data or contact a cluster.

The density construction is a local-concentration graph proxy, not a normalized
PDF. Region review summaries describe the existing reviewed posts, not unbiased
regional population averages. Starts/subsamples measure algorithmic stability,
not sampling uncertainty or temporal change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import signal
import sys
import time
from importlib.metadata import version
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from semantic_region_density import fit_density_regions, predict_density_regions
from semantic_region_directional import fit_spherical_kmeans, fit_vmf, predict_spherical, predict_vmf


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unique_rows(rows, key='annotation_id'):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows) or '' in result:
        raise ValueError(f'Duplicate or missing {key}')
    return result


def priority(seed, aid):
    return hashlib.sha256(f'{seed}|{aid}'.encode()).hexdigest()


def load_metadata(config, data_root):
    paths = {key: data_root / config['input'][key]
             for key in ('index', 'manifest', 'roles', 'human_reviews')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')
    for key, expected in config.get('input_sha256', {}).items():
        if sha256(paths[key]) != expected:
            raise ValueError(f'Frozen input hash mismatch: {key}')
    index = unique_rows(read_csv(paths['index']))
    roles = unique_rows(read_csv(paths['roles']))
    reviews = unique_rows(read_csv(paths['human_reviews']))
    if set(index) != set(roles):
        raise ValueError('Index and historical data roles do not cover the same posts')
    if set(row['sample_role'] for row in roles.values()) - {'training', 'heldout_historical'}:
        raise ValueError('Unrecognized historical data role')
    for aid, row in reviews.items():
        if aid not in index or row['post_id'] != index[aid]['post_id'] or row['text_sha256'] != index[aid]['text_sha256']:
            raise ValueError('Human-review identity/text hash does not match the embedding index')
        for rater in ('derek', 'arya'):
            if row[f'{rater}_personal_req_score'] not in ('0', '1', '2', '3'):
                raise ValueError('Missing or invalid existing personal-context score')
    manifest = json.loads(paths['manifest'].read_text())
    groups = defaultdict(list)
    for aid, row in index.items():
        groups[(roles[aid]['sample_role'], row['subreddit'])].append(aid)
    selected = {}
    communities = sorted({row['subreddit'] for row in index.values()})
    for role, key in [('training', 'training_per_community'), ('heldout_historical', 'evaluation_per_community')]:
        chosen = []
        requested = config['sampling'][key]
        if not isinstance(requested, int) or requested < 1:
            raise ValueError('Posts per community must be positive integers')
        for community in communities:
            ids = groups.get((role, community), [])
            if len(ids) < requested:
                raise ValueError(f'{community} lacks {requested} posts in role {role}')
            chosen.extend(sorted(ids, key=lambda aid: priority(config['sampling']['seed'], aid))[:requested])
        selected[role] = chosen
    if not selected['training'] or not selected['heldout_historical']:
        raise ValueError('Both training and evaluation posts are required')
    return index, roles, reviews, manifest, selected, paths


def load_vectors(index, ids, manifest, shard_dir, dimension):
    """Authenticate each used shard before reading vectors; never allow pickle."""
    grouped = defaultdict(list)
    for aid in ids:
        grouped[index[aid]['shard']].append(aid)
    missing = [name for name in sorted(grouped) if not (shard_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} required embedding shards missing from {shard_dir}; first: {missing[0]}')
    vectors, hashes = {}, {}
    for name, aids in sorted(grouped.items()):
        if Path(name).name != name:
            raise ValueError('Shard index must use plain filenames')
        path = shard_dir / name
        digest = sha256(path)
        if digest != manifest.get('completed_shards', {}).get(name):
            raise ValueError(f'Shard hash mismatch: {name}')
        hashes[name] = digest
        with np.load(path, allow_pickle=False) as shard:
            if str(shard['cache_key'].item()) != manifest['cache_key']:
                raise ValueError(f'Encoder cache key mismatch: {name}')
            shard_ids = shard['annotation_ids'].astype(str)
            shard_text_hashes = shard['text_sha256'].astype(str)
            shard_vectors = shard['embeddings']
            if shard_vectors.shape != (len(shard_ids), dimension) or shard_text_hashes.shape != shard_ids.shape:
                raise ValueError(f'Inconsistent arrays in shard {name}')
            for aid in aids:
                row = int(index[aid]['shard_row'])
                if not 0 <= row < len(shard_ids) or shard_ids[row] != aid or shard_text_hashes[row] != index[aid]['text_sha256']:
                    raise ValueError(f'Post identity/hash mismatch in shard {name}')
                x = np.asarray(shard_vectors[row], dtype=np.float64)
                if x.shape != (dimension,) or not np.isfinite(x).all() or not np.isclose(np.linalg.norm(x), 1, atol=1e-4, rtol=1e-4):
                    raise ValueError(f'Invalid embedding for {aid}')
                vectors[aid] = x / np.linalg.norm(x)
    return vectors, hashes


def model_specs(config):
    settings = config['models']
    specs = []
    for family in ('spherical_kmeans', 'vmf'):
        for k in settings['k']:
            base = f'{family}_k{k}'
            for seed in settings['seeds']:
                specs.append(dict(base=base, family=family, k=k, seed=seed, variant=f'seed{seed}', subsample=False))
            specs.append(dict(base=base, family=family, k=k, seed=settings['seeds'][0], variant='training_subsample', subsample=True))
    for k in settings['density_neighbors']:
        for q in settings['density_quantiles']:
            base = f'density_nn{k}_q{q:g}'
            for subsample in (False, True):
                specs.append(dict(base=base, family='density', neighbors=k, quantile=q,
                                  seed=settings['seeds'][0], variant='training_subsample' if subsample else 'reference', subsample=subsample))
    return specs


def predict(family, X, model):
    if family == 'spherical_kmeans':
        labels = predict_spherical(X, model['centers'])
        confidence = np.max(X @ model['centers'].T, axis=1)
        return labels, confidence, None, None
    if family == 'vmf':
        labels, probabilities, log_density = predict_vmf(X, model)
        return labels, probabilities.max(axis=1), probabilities, log_density
    labels = predict_density_regions(X, model)
    return labels, None, None, None


def partition_comparison(reference, other):
    common = (reference >= 0) & (other >= 0)
    first = np.unique(reference[common])
    second = np.unique(other[common])
    # A single group or all-noise output is not evidence of useful stability.
    ari = float(adjusted_rand_score(reference[common], other[common])) if len(first) > 1 and len(second) > 1 else None
    return {'joint_assignment_fraction': float(common.mean()), 'jointly_assigned_posts': int(common.sum()),
            'reference_regions_on_joint_posts': len(first), 'other_regions_on_joint_posts': len(second),
            'adjusted_rand_joint_assigned': ari}


def align_labels(reference, other, labels_to_map):
    common = (reference >= 0) & (other >= 0)
    a, b = np.unique(reference[common]), np.unique(other[common])
    mapped = np.full(len(labels_to_map), -2, dtype=int)
    mapped[labels_to_map < 0] = -1
    if len(a) and len(b):
        table = np.zeros((len(a), len(b)), dtype=int)
        np.add.at(table, (np.searchsorted(a, reference[common]), np.searchsorted(b, other[common])), 1)
        aa, bb = linear_sum_assignment(-table)
        for i, j in zip(aa, bb):
            if table[i, j] > 0:
                mapped[labels_to_map == b[j]] = a[i]
    return mapped


def build_report(output, status, summaries, comparisons, input_summary):
    lines = ['# Small semantic-region comparison', '', f'**Status:** {status}', '',
             f"Training: {input_summary['training_posts']}; comparison: {input_summary['evaluation_posts']}; existing reviewed posts: {input_summary['human_review_posts']}.", '',
             '| Fit | Regions in comparison posts | Assigned | Largest region share | Converged | Seconds |',
             '|---|---:|---:|---:|---|---:|']
    for row in summaries:
        lines.append(f"| {row['fit']} | {row['evaluation_regions']} | {row['evaluation_coverage']:.1%} | {row['largest_region_share_all_evaluation']:.1%} | {row['converged']} | {row['seconds']:.1f} |")
    lines += ['', 'See `stability.csv` for repeated-start and training-subsample comparisons, '
              '`region_review_summary.csv` for the separate human ratings by region, and '
              '`review_assignments_restricted.csv` for the reviewed-post assignments.', '',
              'The density graph uses a neighborhood concentration proxy and can reject posts. '
              'Its training level set and its frozen prediction rule have different coverage. '
              'vMF components are candidate groups, not automatically density peaks. '
              'Inspect convergence and concentration-cap hits before interpreting a fit.', '',
              'Human-score summaries describe the reviewed subset, not representative regional population means. '
              'Stability uses post subsampling within communities and is not an inferential confidence interval. '
              'These results alone do not select a substantively meaningful winner. '
              'No temporal comparison, new human scoring, or STM fitting is performed.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def run(config, data_root, shard_dir, output):
    index, roles, reviews, manifest, selected, paths = load_metadata(config, data_root)
    train_ids = selected['training']
    eval_ids = selected['heldout_historical']
    review_ids = sorted(reviews)
    all_ids = sorted(set(train_ids) | set(eval_ids) | set(review_ids))
    vectors, shard_hashes = load_vectors(index, all_ids, manifest, shard_dir, config['input']['dimension'])
    X = np.asarray([vectors[i] for i in train_ids])
    E = np.asarray([vectors[i] for i in eval_ids])
    H = np.asarray([vectors[i] for i in review_ids])
    fraction = config['sampling']['stability_training_fraction']
    if not 0 < fraction < 1:
        raise ValueError('Stability training fraction must lie strictly between zero and one')
    group_indexes = defaultdict(list)
    for j, aid in enumerate(train_ids):
        group_indexes[index[aid]['subreddit']].append(j)
    subsample = []
    for group in group_indexes.values():
        keep = max(1, int(np.floor(len(group) * fraction)))
        subsample.extend(sorted(group, key=lambda j: priority(config['sampling']['seed'] + 1, train_ids[j]))[:keep])
    variants = {False: np.arange(len(X)), True: np.asarray(subsample)}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'fits').mkdir()
    write_json(output / 'config.json', config)
    input_summary = {'training_posts': len(X), 'evaluation_posts': len(E), 'human_review_posts': len(H),
                     'training_communities': len(group_indexes), 'stability_training_posts': len(subsample),
                     'dimension': X.shape[1], 'input_sha256': {key: sha256(path) for key, path in paths.items()},
                     'embedding_cache_key': manifest['cache_key'], 'verified_shard_sha256': shard_hashes,
                     'runtime': {'python': sys.version, **{name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')}},
                     'script_sha256': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('semantic_region_directional.py'), Path(__file__).with_name('semantic_region_density.py')]}}
    write_json(output / 'inputs.json', input_summary)
    train_set, eval_set, review_set = set(train_ids), set(eval_ids), set(review_ids)
    write_csv(output / 'selected_posts_restricted.csv', [dict(annotation_id=i, post_id=index[i]['post_id'],
              subreddit=index[i]['subreddit'], year_month=index[i]['year_month'], text_sha256=index[i]['text_sha256'],
              historical_role=roles[i]['sample_role'], pilot_training=int(i in train_set),
              pilot_evaluation=int(i in eval_set), existing_human_review=int(i in review_set)) for i in all_ids])
    summaries, comparisons, region_reviews, review_assignments = [], [], [], []
    reference = {}
    status, error = 'running', None
    settings = config['models']
    started = time.monotonic()
    limit = config['resources']['fit_wall_limit_seconds']
    if not isinstance(limit, (int, float)) or not np.isfinite(limit) or limit <= 0:
        raise ValueError('Fit wall-time limit must be positive and finite')
    write_json(output / 'status.json', {'status': status, 'completed_fits': 0, 'planned_fits': len(model_specs(config))})
    def timed_out(signum, frame):
        raise TimeoutError('Declared fit wall-time limit reached')
    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        with threadpool_limits(limits=config['resources']['threads']):
            for spec in model_specs(config):
                fit_id = spec['base'] + '_' + spec['variant']
                print(f'Fitting {fit_id}', flush=True)
                t0 = time.monotonic()
                positions = variants[spec['subsample']]
                fit_X = X[positions]
                if spec['family'] == 'spherical_kmeans':
                    model = fit_spherical_kmeans(fit_X, spec['k'], spec['seed'], max_iter=settings['max_iter'], tol=settings['tolerance'])
                elif spec['family'] == 'vmf':
                    model = fit_vmf(fit_X, spec['k'], spec['seed'], kappa_max=settings['vmf_kappa_max'], max_iter=settings['max_iter'], tol=settings['tolerance'])
                else:
                    model = fit_density_regions(fit_X, spec['neighbors'], spec['quantile'], settings['density_min_region_size'])
                labels, confidence, probabilities, log_density = predict(spec['family'], E, model)
                human_labels, human_confidence, human_probabilities, _ = predict(spec['family'], H, model)
                # For reviewed posts actually used in this fit, retain training
                # level-set membership rather than applying its extension rule.
                position_by_id = {train_ids[j]: k for k, j in enumerate(positions)}
                for j, aid in enumerate(review_ids):
                    if aid in position_by_id:
                        human_labels[j] = model['labels'][position_by_id[aid]]
                accepted = labels >= 0
                counts = Counter(labels[accepted].tolist())
                fit_row = {'fit': fit_id, 'family': spec['family'], 'variant': spec['variant'],
                           'training_posts': len(fit_X), 'training_regions': len(set(model['labels'][model['labels'] >= 0])),
                           'training_coverage': float(np.mean(model['labels'] >= 0)), 'evaluation_regions': len(counts),
                           'evaluation_coverage': float(accepted.mean()), 'largest_region_share_all_evaluation': max(counts.values(), default=0) / len(E),
                           'human_review_coverage': float(np.mean(human_labels >= 0)), 'iterations': model.get('iterations'),
                           'converged': model.get('converged', 'deterministic'), 'seconds': time.monotonic() - t0}
                if spec['family'] == 'vmf':
                    fit_row.update(mean_evaluation_log_density=float(log_density.mean()),
                                   concentration_cap_hits=int(np.sum(np.isclose(model['kappas'], settings['vmf_kappa_max']))),
                                   mean_max_membership=float(confidence.mean()))
                if spec['family'] == 'spherical_kmeans':
                    fit_row['mean_evaluation_cosine_distance'] = float(np.mean(1 - confidence))
                if spec['family'] == 'density':
                    fit_row['threshold_radius'] = model['threshold_radius']
                arrays = {'training_ids': np.asarray([train_ids[j] for j in positions]), 'training_labels': model['labels'],
                          'evaluation_ids': np.asarray(eval_ids), 'evaluation_labels': labels,
                          'review_ids': np.asarray(review_ids), 'review_labels': human_labels}
                for key in ('centers', 'weights', 'kappas', 'core_distances', 'neighbor_indices', 'objective_history', 'log_likelihood_history'):
                    if key in model:
                        arrays[key] = model[key]
                if probabilities is not None:
                    arrays['evaluation_memberships'] = probabilities.astype(np.float32)
                    arrays['review_memberships'] = human_probabilities.astype(np.float32)
                np.savez_compressed(output / 'fits' / (fit_id + '.npz'), **arrays)
                write_json(output / 'fits' / (fit_id + '.json'), {'settings': spec, 'summary': fit_row,
                           'density_prediction_rule': model.get('extension_rule'), 'density_proxy': model.get('density_proxy')})
                base = reference.get(spec['base'])
                mapped = None
                if base is not None:
                    comparisons.append(dict(reference_fit=base['fit'], other_fit=fit_id, **partition_comparison(base['labels'], labels)))
                    mapped = align_labels(base['labels'], labels, human_labels)
                else:
                    reference[spec['base']] = {'fit': fit_id, 'labels': labels.copy(), 'human_labels': human_labels.copy()}
                for j, aid in enumerate(review_ids):
                    row = {'fit': fit_id, 'annotation_id': aid, 'historical_role': roles[aid]['sample_role'],
                           'used_in_this_fit': int(aid in position_by_id), 'region': int(human_labels[j]),
                           'derek_personal_req_score': reviews[aid]['derek_personal_req_score'],
                           'arya_personal_req_score': reviews[aid]['arya_personal_req_score']}
                    if mapped is not None:
                        valid = base['human_labels'][j] >= 0 and human_labels[j] >= 0 and mapped[j] >= 0
                        row['same_region_as_reference_if_both_assigned'] = int(mapped[j] == base['human_labels'][j]) if valid else ''
                        row['reference_comparison_status'] = 'aligned' if valid else 'rejected_or_unaligned'
                    review_assignments.append(row)
                for region in sorted(set(model['labels']) | set(labels) | set(human_labels)):
                    mask = human_labels == region
                    review_a = np.asarray([int(reviews[i]['derek_personal_req_score']) for i in review_ids])[mask]
                    review_b = np.asarray([int(reviews[i]['arya_personal_req_score']) for i in review_ids])[mask]
                    region_reviews.append({'fit': fit_id, 'region': int(region), 'rejected': int(region < 0),
                            'training_posts': int(np.sum(model['labels'] == region)), 'evaluation_posts': int(np.sum(labels == region)),
                            'reviewed_posts': int(mask.sum()), 'reviewed_posts_used_in_fit': sum(aid in position_by_id for aid, keep in zip(review_ids, mask) if keep),
                            'derek_reviewed_mean': float(review_a.mean()) if len(review_a) else None,
                            'arya_reviewed_mean': float(review_b.mean()) if len(review_b) else None,
                            'rater_exact_agreement_fraction': float(np.mean(review_a == review_b)) if len(review_a) else None})
                summaries.append(fit_row)
                write_csv(output / 'fit_summary.csv', summaries)
                write_json(output / 'status.json', {'status': 'running', 'completed_fits': len(summaries), 'planned_fits': len(model_specs(config))})
                print(f'Completed {fit_id}: {len(counts)} regions, {accepted.mean():.1%} assigned', flush=True)
            status = 'complete'
    except KeyboardInterrupt:
        status, error = 'interrupted', 'KeyboardInterrupt'
    except Exception as exc:
        status = 'partial_time_limit' if isinstance(exc, TimeoutError) else 'failed'
        error = f'{type(exc).__name__}: {exc}'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        write_csv(output / 'stability.csv', comparisons)
        write_csv(output / 'region_review_summary.csv', region_reviews)
        write_csv(output / 'review_assignments_restricted.csv', review_assignments)
        write_json(output / 'status.json', {'status': status, 'error': error, 'completed_fits': len(summaries),
                   'planned_fits': len(model_specs(config)), 'fit_elapsed_seconds': time.monotonic() - started,
                   'finished_utc': datetime.now(timezone.utc).isoformat(), 'new_human_scores': 0})
        build_report(output, status, summaries, comparisons, input_summary)
    return 0 if status == 'complete' else (130 if status == 'interrupted' else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'semantic_region_pilot_config.json')
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--shards', type=Path, help='Override the local embedding-shard directory')
    parser.add_argument('--output', type=Path, help='A new output directory; existing directories are never overwritten')
    parser.add_argument('--preflight', action='store_true', help='Check metadata and required shard availability without fitting')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    shard_dir = args.shards or args.data_root / config['input']['shards']
    try:
        if args.preflight:
            index, roles, reviews, manifest, selected, paths = load_metadata(config, args.data_root)
            ids = set(selected['training']) | set(selected['heldout_historical']) | set(reviews)
            names = {index[i]['shard'] for i in ids}
            missing = sorted(name for name in names if not (shard_dir / name).is_file())
            print(json.dumps({'status': 'missing_embeddings' if missing else 'ready_for_authenticated_load',
                       'training_posts': len(selected['training']), 'evaluation_posts': len(selected['heldout_historical']),
                       'reviewed_posts': len(reviews), 'required_shards': len(names), 'missing_shards': len(missing),
                       'shard_directory': str(shard_dir), 'planned_fits': len(model_specs(config))}, indent=2))
            return 2 if missing else 0
        if args.output is None:
            parser.error('--output is required for fitting')
        if args.output.exists():
            raise FileExistsError(f'Output directory already exists: {args.output}')
        return run(config, args.data_root, shard_dir, args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f'Cannot run comparison: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
