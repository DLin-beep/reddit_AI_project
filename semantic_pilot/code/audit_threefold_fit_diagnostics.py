#!/usr/bin/env python3
"""Inspect authenticated completed pilot fits without fitting or changing files.

This supplemental audit reports convergence histories, component occupancy,
assignment confidence, and density coverage. It does not change the frozen
threefold grid or classify any region as an availability response.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ARTIFACTS = {'fit.json', 'model.npz', 'predictions.npz', 'review_assignments_restricted.csv'}
MONOTONICITY_TOLERANCE = 1e-7  # Existing vMF kernel's average-LL decrease tolerance.


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _finite_vector(values, name, nonempty=True):
    array = np.asarray(values)
    if array.ndim != 1 or (nonempty and not len(array)) or not np.isfinite(array).all():
        raise ValueError(f'Invalid {name} array')
    return array


def _history(values, divisor=1., improvement_sign=1.):
    array = _finite_vector(values, 'objective history')
    improvements = improvement_sign * np.diff(array.astype(float)) / divisor
    return {
        'history_points': len(array),
        'final_improvement': float(improvements[-1]) if len(improvements) else None,
        'last_ten_improvements': improvements[-10:].tolist(),
        'mean_last_ten_improvements': float(np.mean(improvements[-10:])) if len(improvements) else None,
        'minimum_improvement': float(improvements.min()) if len(improvements) else None,
        'monotonicity_tolerance': MONOTONICITY_TOLERANCE,
        'decreases_exceeding_tolerance': int(np.sum(improvements < -MONOTONICITY_TOLERANCE)),
    }


def _authenticate_task(root, spec, inputs, config_hash):
    directory = root / 'tasks' / f"task_{spec['task_id']:03d}"
    status_path = directory / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'missing'}
    if status.get('status') != 'complete':
        return None, status.get('status', 'unknown')
    if status.get('task_id') != spec['task_id'] or status.get('fit_id') != spec['fit_id']:
        raise ValueError('Completed task identity differs from the planned task')
    if status.get('config_sha256') != config_hash or status.get('input_fingerprint') != inputs.get('input_fingerprint'):
        raise ValueError('Completed task differs from committed configuration or inputs')
    hashes = status.get('artifact_sha256', {})
    if set(hashes) != ARTIFACTS:
        raise ValueError('Completed task has an unexpected artifact inventory')
    # Verify all committed artifacts before reading their content.
    for name, expected in hashes.items():
        artifact = directory / name
        if artifact.is_symlink() or not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"Completed task {spec['task_id']} has a missing or modified {name}")
    fit = json.loads((directory / 'fit.json').read_text())
    if fit.get('task') != spec or fit.get('config_sha256') != config_hash or fit.get('input_fingerprint') != inputs.get('input_fingerprint'):
        raise ValueError('Fit artifact identity differs from the committed task')
    return (directory, fit), 'complete'


def _diagnose(directory, fit, config):
    spec, scalars = fit['task'], fit.get('model_scalars', {})
    with np.load(directory / 'model.npz', allow_pickle=False) as source:
        model = {key: source[key] for key in (
            'labels', 'training_ids', 'component_sizes', 'effective_component_sizes',
            'weights', 'kappas', 'log_likelihood_history', 'objective_history', 'core_distances',
        ) if key in source.files}
    labels = _finite_vector(model['labels'], 'training labels')
    if not np.issubdtype(labels.dtype, np.integer) or len(labels) != len(model['training_ids']):
        raise ValueError('Training labels and IDs have inconsistent dimensions')
    n_train = len(labels)
    assigned = labels >= 0
    counts = Counter(labels[assigned].tolist())
    settings = config['models']
    row = {
        'task_id': spec['task_id'], 'fit_id': spec['fit_id'], 'family': spec['family'],
        'training_folds': ''.join(spec['train_folds']), 'training_posts': n_train,
        'training_regions_with_hard_assignments': len(counts),
        'iterations': scalars.get('iterations'), 'converged': scalars.get('converged'),
        'maximum_iterations': settings['max_iter'], 'configured_tolerance': settings['tolerance'],
        'fit_seconds': fit.get('fit_seconds'),
    }
    evaluation = []
    for summary in fit['summaries']:
        evaluation.append({key: summary.get(key) for key in (
            'evaluation_fold', 'evaluation_posts', 'evaluation_regions', 'evaluation_coverage',
            'largest_region_share_all_evaluation', 'mean_max_membership',
            'mean_evaluation_log_density', 'mean_evaluation_cosine_distance', 'concentration_cap_hits',
        ) if key in summary})
    row['evaluation'] = evaluation
    if spec['family'] in ('vmf', 'spherical_kmeans'):
        k = spec['k']
        if not np.all(assigned) or (len(counts) and max(counts) >= k):
            raise ValueError('Directional-model training labels exceed the configured component range')
        hard_counts = np.bincount(labels, minlength=k)
        if 'component_sizes' in model and not np.array_equal(hard_counts, model['component_sizes']):
            raise ValueError('Stored hard component sizes differ from training assignments')
        row.update(requested_components=k, empty_hard_components=int(np.sum(hard_counts == 0)),
                   minimum_hard_component_posts=int(hard_counts.min()),
                   maximum_hard_component_posts=int(hard_counts.max()),
                   minimum_hard_component_share=float(hard_counts.min() / n_train),
                   maximum_hard_component_share=float(hard_counts.max() / n_train))
    if spec['family'] == 'vmf':
        history = _history(model['log_likelihood_history'], divisor=n_train)
        row['average_log_likelihood_history'] = history
        row['final_average_log_likelihood'] = float(model['log_likelihood_history'][-1] / n_train)
        row['final_improvement_above_configured_tolerance'] = (
            history['final_improvement'] > settings['tolerance'] if history['final_improvement'] is not None else None)
        kappas = _finite_vector(model['kappas'], 'kappas')
        effective = _finite_vector(model['effective_component_sizes'], 'effective component counts')
        weights = _finite_vector(model['weights'], 'mixture weights')
        if len(kappas) != spec['k'] or len(effective) != spec['k'] or len(weights) != spec['k']:
            raise ValueError('vMF component parameter dimensions disagree')
        if np.any(kappas < 0) or np.any(effective < 0) or np.any(weights < 0) or not np.isclose(effective.sum(), n_train) or not np.isclose(weights.sum(), 1):
            raise ValueError('vMF component masses or concentrations are invalid')
        row.update(minimum_kappa=float(kappas.min()), maximum_kappa=float(kappas.max()),
                   concentration_cap=settings['vmf_kappa_max'],
                   concentration_cap_hits=int(np.sum(np.isclose(kappas, settings['vmf_kappa_max']))),
                   minimum_effective_component_posts=float(effective.min()),
                   maximum_effective_component_posts=float(effective.max()),
                   minimum_effective_component_share=float(effective.min() / n_train),
                   minimum_mixture_weight=float(weights.min()),
                   initialization_iterations=scalars.get('initialization_iterations'))
        # The stored confidence vector is each post's maximum soft membership;
        # reading it avoids loading the full n-by-k responsibility matrices.
        with np.load(directory / 'predictions.npz', allow_pickle=False) as predictions:
            for summary in evaluation:
                fold = summary['evaluation_fold']
                confidence = _finite_vector(predictions[f'evaluation_confidence_{fold}'], 'maximum-membership confidence')
                if len(confidence) != len(predictions[f'evaluation_ids_{fold}']) or np.any(confidence < -1e-6) or np.any(confidence > 1 + 1e-6):
                    raise ValueError('Invalid maximum-membership confidence vector')
                summary['maximum_membership_quantiles_05_50_95'] = np.quantile(confidence, [.05, .5, .95]).tolist()
                summary['fraction_maximum_membership_at_least_095'] = float(np.mean(confidence >= .95))
                summary['fraction_maximum_membership_at_least_099'] = float(np.mean(confidence >= .99))
    elif spec['family'] == 'spherical_kmeans':
        row['mean_cosine_distance_history'] = _history(model['objective_history'], improvement_sign=-1.)
        row['final_mean_cosine_distance'] = float(model['objective_history'][-1])
    elif spec['family'] == 'density':
        radii = _finite_vector(model['core_distances'], 'core distances')
        if len(radii) != n_train or np.any(radii < 0):
            raise ValueError('Invalid density core-radius array')
        row.update(converged='deterministic', n_neighbors=spec['neighbors'], density_quantile=spec['quantile'],
                   minimum_region_size=settings['density_min_region_size'],
                   retained_posts=scalars.get('n_retained'), retained_fraction=scalars.get('retained_fraction'),
                   training_coverage=float(assigned.mean()),
                   largest_training_region_posts=max(counts.values(), default=0),
                   largest_training_region_share_all_posts=max(counts.values(), default=0) / n_train,
                   largest_training_region_share_assigned_posts=max(counts.values(), default=0) / int(assigned.sum()) if assigned.any() else None,
                   threshold_radius=scalars.get('threshold_radius'),
                   core_radius_quantiles_05_50_95=np.quantile(radii, [.05, .5, .95]).tolist(),
                   minimum_core_radius=float(radii.min()), maximum_core_radius=float(radii.max()),
                   neighbor_cache_hit=scalars.get('neighbor_cache_hit'),
                   neighbors_cached=scalars.get('neighbors_cached'), requested_threads=scalars.get('threads'))
    else:
        raise ValueError('Unrecognized model family')
    return row


def audit(results, task_ids=None):
    root = Path(results)
    config = json.loads((root / 'config.json').read_text())
    inputs = json.loads((root / 'inputs.json').read_text())
    specs = json.loads((root / 'tasks.json').read_text())
    if not isinstance(inputs.get('input_fingerprint'), str) or not inputs['input_fingerprint']:
        raise ValueError('Committed input fingerprint is missing')
    if len({spec['task_id'] for spec in specs}) != len(specs):
        raise ValueError('Planned task IDs are not unique')
    for spec in specs:
        if isinstance(spec['task_id'], bool) or not isinstance(spec['task_id'], int) or spec['task_id'] < 0:
            raise ValueError('Planned task IDs must be nonnegative integers')
    if task_ids is not None:
        selected = set(task_ids)
        if selected - {spec['task_id'] for spec in specs}:
            raise ValueError('Requested task IDs are not in the committed plan')
        specs = [spec for spec in specs if spec['task_id'] in selected]
    rows, pending = [], []
    config_hash = _digest(config)
    for spec in specs:
        task, status = _authenticate_task(root, spec, inputs, config_hash)
        if task is None:
            pending.append({'task_id': spec['task_id'], 'status': status})
        else:
            rows.append(_diagnose(*task, config))
    return {
        'schema_version': 1,
        'selected_tasks': len(specs), 'completed_tasks_audited': len(rows),
        'pending_tasks': pending,
        'summary': {
            'completed_by_family': dict(Counter(row['family'] for row in rows)),
            'nonconverged_directional_fits': sum(row['converged'] is False for row in rows),
            'vmf_fits_with_likelihood_decrease_above_tolerance': sum(
                row.get('average_log_likelihood_history', {}).get('decreases_exceeding_tolerance', 0) > 0 for row in rows),
            'density_fits_with_at_most_one_assigned_region': sum(
                row['family'] == 'density' and row['training_regions_with_hard_assignments'] <= 1 for row in rows),
        },
        'fits': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--task-id', type=int, action='append', help='Audit only selected task IDs; may be repeated')
    args = parser.parse_args(argv)
    try:
        result = audit(args.results, args.task_id)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f'Cannot audit completed fits: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
