#!/usr/bin/env python3
"""Describe authenticated three-fold semantic-region fits without selecting a winner.

All clustering comparisons exclude rejected posts from the ARI and show the
rejection/coverage change separately. Region correspondence uses overlap on the
same evaluation posts, never equality of arbitrary numeric labels. Historical
personal-context annotations remain separate by rater and are descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows, empty_fields=('status',)):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _labels(values):
    result = np.asarray(values)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError('Assignments must be one-dimensional integer arrays')
    return result.astype(np.int64, copy=False)


def compare_partitions(left, right):
    """Return coverage, nondegenerate joint ARI, and bidirectional region overlaps.

    Region sizes use the entire common evaluation sample, including members
    rejected by the other map. This prevents partial coverage from inflating
    Jaccard overlap. One-to-one matches maximize total Jaccard, while unrestricted
    best matches and all nonzero edges expose possible splits and merges.
    """
    left, right = _labels(left), _labels(right)
    if len(left) != len(right) or not len(left):
        raise ValueError('Comparisons require the same nonempty evaluation sample')
    lc = Counter(int(x) for x in left if x >= 0)
    rc = Counter(int(x) for x in right if x >= 0)
    la, ra = sorted(lc), sorted(rc)
    joint = (left >= 0) & (right >= 0)
    nl, nr = len(set(left[joint])), len(set(right[joint]))
    flags = []
    if not lc:
        flags.append('left_all_rejected')
    if not rc:
        flags.append('right_all_rejected')
    if len(lc) == 1:
        flags.append('left_single_region')
    if len(rc) == 1:
        flags.append('right_single_region')
    if joint.sum() < 2:
        flags.append('fewer_than_two_jointly_assigned_posts')
    if nl < 2 or nr < 2:
        flags.append('joint_partition_degenerate')
    ari = float(adjusted_rand_score(left[joint], right[joint])) if joint.sum() >= 2 and nl >= 2 and nr >= 2 else None
    summary = {
        'evaluation_posts': len(left),
        'left_assigned_posts': sum(lc.values()), 'right_assigned_posts': sum(rc.values()),
        'left_coverage': sum(lc.values()) / len(left), 'right_coverage': sum(rc.values()) / len(left),
        'jointly_assigned_posts': int(joint.sum()), 'joint_assignment_fraction': float(joint.mean()),
        'left_only_assigned_posts': int(((left >= 0) & (right < 0)).sum()),
        'right_only_assigned_posts': int(((right >= 0) & (left < 0)).sum()),
        'both_rejected_posts': int(((left < 0) & (right < 0)).sum()),
        'left_regions': len(lc), 'right_regions': len(rc),
        'left_regions_on_joint_posts': nl, 'right_regions_on_joint_posts': nr,
        'adjusted_rand_joint_assigned': ari, 'degenerate_flags': ';'.join(flags),
    }
    table = np.zeros((len(la), len(ra)), dtype=np.int64)
    if joint.any():
        np.add.at(table, (np.searchsorted(la, left[joint]), np.searchsorted(ra, right[joint])), 1)
    unions = np.asarray([lc[x] for x in la])[:, None] + np.asarray([rc[x] for x in ra])[None, :] - table
    jaccard = np.divide(table, unions, out=np.zeros_like(table, dtype=float), where=unions > 0)
    one_left, one_right = {}, {}
    if len(la) and len(ra):
        ii, jj = linear_sum_assignment(-jaccard)
        for i, j in zip(ii, jj):
            if table[i, j] > 0:
                one_left[la[i]], one_right[ra[j]] = ra[j], la[i]
    edges = []
    for i, j in zip(*np.nonzero(table)):
        edges.append({
            'left_region': la[i], 'right_region': ra[j],
            'left_region_posts': lc[la[i]], 'right_region_posts': rc[ra[j]],
            'overlap_posts': int(table[i, j]),
            'left_fraction_in_right_region': float(table[i, j] / lc[la[i]]),
            'right_fraction_in_left_region': float(table[i, j] / rc[ra[j]]),
            'jaccard': float(jaccard[i, j]),
            'one_to_one_match': int(one_left.get(la[i]) == ra[j]),
        })
    region_rows = []
    for side, own, other, counts, other_labels, one, cells in (
        ('left', la, ra, lc, right, one_left, table),
        ('right', ra, la, rc, left, one_right, table.T),
    ):
        own_labels = left if side == 'left' else right
        for i, region in enumerate(own):
            mask = own_labels == region
            candidates = []
            for j in np.flatnonzero(cells[i]):
                overlap = int(cells[i, j])
                other_n = (rc if side == 'left' else lc)[other[j]]
                candidates.append({'region': other[j], 'posts': overlap,
                                   'source_fraction': overlap / counts[region],
                                   'target_fraction': overlap / other_n,
                                   'jaccard': overlap / (counts[region] + other_n - overlap)})
            candidates.sort(key=lambda row: (-row['jaccard'], -row['posts'], row['region']))
            best = candidates[0] if candidates else {}
            matched = next((row for row in candidates if row['region'] == one.get(region)), {})
            region_rows.append({
                'side': side, 'region': region, 'region_posts': counts[region],
                'region_share_all_evaluation': counts[region] / len(left),
                'posts_rejected_by_other': int((other_labels[mask] < 0).sum()),
                'fraction_rejected_by_other': float((other_labels[mask] < 0).mean()),
                'overlapping_other_regions': len(candidates),
                'best_overlap_region': best.get('region'), 'best_jaccard': best.get('jaccard', 0.),
                'best_overlap_source_fraction': best.get('source_fraction', 0.),
                'best_overlap_target_fraction': best.get('target_fraction', 0.),
                'one_to_one_region': matched.get('region'), 'one_to_one_jaccard': matched.get('jaccard', 0.),
                'overlaps_json': json.dumps(candidates, separators=(',', ':')),
            })
    summary['mean_best_jaccard_left_regions'] = float(np.mean([r['best_jaccard'] for r in region_rows if r['side'] == 'left'])) if la else None
    summary['mean_best_jaccard_right_regions'] = float(np.mean([r['best_jaccard'] for r in region_rows if r['side'] == 'right'])) if ra else None
    return summary, region_rows, edges


def comparison_kind(a, b):
    """Classify design relationships, without calling nested fits replications."""
    if a['family'] != b['family'] or a['base'] != b['base']:
        return None
    at, bt = frozenset(a['train_folds']), frozenset(b['train_folds'])
    same_seed = a.get('seed') == b.get('seed')
    if at == bt and not same_seed and a['family'] != 'density':
        return 'seed_stability'
    if same_seed and len(at) == len(bt) == 1 and at.isdisjoint(bt):
        return 'independent_fit_reproducibility'
    if same_seed and (at < bt or bt < at) and sorted((len(at), len(bt))) == [1, 2]:
        return 'nested_training_size_consistency'
    return None


def select_examples(ids, labels, centrality=None, confidence=None, *, seed=20260914, counts=(3, 3, 2)):
    """Choose distinct central, reproducibly random and lower-confidence posts.

    Confidence tails are descriptive within a model and region, not uncertainty
    calibrated across methods. Random examples are chosen first so their sample
    does not depend on the central/low-confidence selection.
    """
    ids, labels = np.asarray(ids).astype(str), _labels(labels)
    if len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError('Example IDs must be unique and match assignments')
    centrality = np.asarray(centrality, dtype=float) if centrality is not None else None
    confidence = np.asarray(confidence, dtype=float) if confidence is not None else None
    for values in (centrality, confidence):
        if values is not None and values.shape != labels.shape:
            raise ValueError('Example scores must match assignments')
    selected = []
    for region in sorted(set(labels[labels >= 0])):
        positions = np.flatnonzero(labels == region).tolist()
        random_order = sorted(positions, key=lambda j: hashlib.sha256(f'{seed}|{ids[j]}'.encode()).hexdigest())
        random_positions = random_order[:counts[1]]
        selected_region = [('random', j) for j in random_positions]
        used = set(random_positions)
        if centrality is not None:
            candidates = [j for j in positions if j not in used and np.isfinite(centrality[j])]
            central_positions = sorted(candidates, key=lambda j: (-centrality[j], ids[j]))[:counts[0]]
            selected_region.extend(('central', j) for j in central_positions)
            used.update(central_positions)
        if confidence is not None:
            candidates = [j for j in positions if j not in used and np.isfinite(confidence[j])]
            selected_region.extend(('lower_confidence', j) for j in sorted(candidates, key=lambda j: (confidence[j], ids[j]))[:counts[2]])
        for kind, j in selected_region:
            selected.append({'region': int(region), 'annotation_id': str(ids[j]), 'selection': kind,
                             'evaluation_region_posts': len(positions),
                             'cosine_centrality': float(centrality[j]) if centrality is not None and np.isfinite(centrality[j]) else None,
                             'method_specific_confidence': float(confidence[j]) if confidence is not None and np.isfinite(confidence[j]) else None})
    return selected


def _authenticated_tasks(output, specs):
    tasks, pending = [], []
    if len({spec['task_id'] for spec in specs}) != len(specs):
        raise ValueError('Duplicate task IDs in the run plan')
    inputs_path = output / 'inputs.json'
    committed_inputs = json.loads(inputs_path.read_text()) if inputs_path.exists() else {}
    config = json.loads((output / 'config.json').read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for spec in specs:
        task_id = spec['task_id']
        if not isinstance(task_id, int) or task_id < 0:
            raise ValueError('Task IDs must be nonnegative integers')
        directory = output / 'tasks' / f'task_{task_id:03d}'
        status_path = directory / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'missing'}
        if status.get('status') != 'complete':
            pending.append({'task_id': task_id, 'fit_id': spec['fit_id'], 'status': status.get('status', 'unknown')})
            continue
        if status.get('task_id') != task_id or status.get('fit_id') != spec['fit_id']:
            raise ValueError('Completed task status identity differs from the run plan')
        hashes = status.get('artifact_sha256', {})
        required = {'fit.json', 'model.npz', 'predictions.npz', 'review_assignments_restricted.csv'}
        if not required.issubset(hashes):
            raise ValueError(f'Completed task {task_id} lacks authenticated artifacts')
        for name, expected in hashes.items():
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('Artifact paths must be simple filenames')
            if (directory / name).is_symlink() or _sha256(directory / name) != expected:
                raise ValueError(f'Artifact hash mismatch: task {task_id} / {name}')
        fit = json.loads((directory / 'fit.json').read_text())
        if fit['task'] != spec or fit.get('config_sha256') != config_hash or status.get('config_sha256') != config_hash:
            raise ValueError(f'Task {task_id} does not match the committed plan/configuration')
        if committed_inputs.get('input_fingerprint') and fit.get('input_fingerprint') != committed_inputs['input_fingerprint']:
            raise ValueError(f'Task {task_id} differs from committed run inputs')
        if fit.get('input_fingerprint') != status.get('input_fingerprint'):
            raise ValueError(f'Task {task_id} input fingerprint mismatch')
        prediction = {}
        with np.load(directory / 'predictions.npz', allow_pickle=False) as data:
            for key in data.files:
                if key.endswith('memberships') or '_memberships_' in key:
                    continue
                prediction[key] = np.asarray(data[key])
        arrays = {}
        with np.load(directory / 'model.npz', allow_pickle=False) as data:
            for key in ('weights', 'kappas'):
                if key in data:
                    arrays[key] = np.asarray(data[key])
        tasks.append({'spec': spec, 'fit': fit, 'predictions': prediction, 'model': arrays,
                      'reviews': _csv(directory / 'review_assignments_restricted.csv')})
    return tasks, pending


def _aligned_predictions(a, b, fold):
    ai = a['predictions'][f'evaluation_ids_{fold}'].astype(str)
    bi = b['predictions'][f'evaluation_ids_{fold}'].astype(str)
    if len(set(ai)) != len(ai) or len(set(bi)) != len(bi) or set(ai) != set(bi):
        raise ValueError('Comparisons must use identical, unique held-out evaluation posts')
    order = {aid: j for j, aid in enumerate(bi)}
    return a['predictions'][f'evaluation_labels_{fold}'], b['predictions'][f'evaluation_labels_{fold}'][[order[aid] for aid in ai]]


def _validate_text(aid, posts, texts):
    item = texts.get(aid)
    if item is None:
        return None
    text = item if isinstance(item, str) else item.get('text')
    if not isinstance(text, str):
        raise ValueError(f'Example text missing for {aid}')
    expected = posts.get(aid, {}).get('text_sha256')
    if not expected or hashlib.sha256(text.encode('utf-8')).hexdigest() != expected:
        raise ValueError(f'Example text hash mismatch for {aid}')
    return text


def _review_summary(task, training_labels, eval_labels):
    original = task['reviews']
    groups = defaultdict(list)
    for row in original:
        region = int(row['region'])
        if region >= 0:
            groups[region].append(row)
    regions = sorted(set(training_labels[training_labels >= 0]) | set(eval_labels[eval_labels >= 0]) | set(groups))
    result = []
    for region in regions:
        rows = groups.get(region, [])
        out = {'region': int(region), 'reviewed_posts': len(rows),
               'reviewed_posts_used_in_fit': sum(int(row.get('used_in_fit', row.get('used_in_this_fit', 0))) for row in rows)}
        for rater in ('derek', 'arya'):
            values = []
            for row in rows:
                raw = row.get(f'{rater}_personal_req_score')
                if raw not in ('0', '1', '2', '3', 0, 1, 2, 3):
                    raise ValueError(f'Invalid original {rater} rating')
                values.append(int(raw))
            out[f'{rater}_reviewed_mean'] = float(np.mean(values)) if values else None
            for score in range(4):
                out[f'{rater}_score_{score}_posts'] = values.count(score)
        out['rater_exact_agreement_fraction'] = float(np.mean([row['derek_personal_req_score'] == row['arya_personal_req_score'] for row in rows])) if rows else None
        out['scope'] = 'historically_selected_reviewed_posts_only'
        result.append(out)
    return result


def report_run(run_dir, posts=None, reviews=None, texts=None):
    """Write report artifacts from committed tasks, allowing an explicit partial run.

    `texts` maps annotation IDs to exact original text strings (or dicts with a
    `text` field). Every included text is checked against the embedding index's
    SHA-256. Missing texts are marked unavailable. No source is fetched here.
    """
    output = Path(run_dir)
    config = json.loads((output / 'config.json').read_text())
    specs = json.loads((output / 'tasks.json').read_text())
    posts = posts if posts is not None else {row['annotation_id']: row for row in _csv(output / 'fold_assignments_restricted.csv')}
    texts = texts or {}
    tasks, pending = _authenticated_tasks(output, specs)
    summaries, sizes, examples, reviews_out, review_regions = [], [], [], [], []
    comparison_rows, overlap_regions, edges_out = [], [], []
    minimum = config.get('models', {}).get('density_min_region_size', 30)
    for task in tasks:
        spec, pred = task['spec'], task['predictions']
        fit_id = spec['fit_id']
        training = _labels(pred['training_labels'])
        training_ids = pred['training_ids'].astype(str)
        if len(training) != len(training_ids) or len(set(training_ids)) != len(training_ids):
            raise ValueError('Training assignments do not match unique IDs')
        prefix = {'fit_id': fit_id, 'family': spec['family'], 'base': spec['base'],
                  'seed': spec.get('seed'), 'train_folds': '+'.join(spec['train_folds'])}
        summaries.extend({**prefix, 'training_posts': len(training), **row,
                          'density_k_over_training_n': spec['neighbors'] / len(training) if spec['family'] == 'density' else None}
                         for row in task['fit']['summaries'])
        if reviews is not None:
            for row in task['reviews']:
                aid = row['annotation_id']
                if aid not in reviews or any(str(row[f'{r}_personal_req_score']) != str(reviews[aid][f'{r}_personal_req_score']) for r in ('derek', 'arya')):
                    raise ValueError('Stored review assignment differs from original annotations')
        reviews_out.extend({**prefix, **row} for row in task['reviews'])
        all_eval = []
        for fold in spec['eval_folds']:
            ids = pred[f'evaluation_ids_{fold}'].astype(str)
            labels = _labels(pred[f'evaluation_labels_{fold}'])
            if set(ids) & set(training_ids):
                raise ValueError('Evaluation posts overlap this fit\'s training posts')
            if fold in spec['train_folds']:
                raise ValueError('Evaluation fold is included in training folds')
            all_eval.extend(labels.tolist())
            tr, ev = Counter(training[training >= 0].tolist()), Counter(labels[labels >= 0].tolist())
            for region in sorted(set(tr) | set(ev)):
                weights = task['model'].get('weights')
                kappas = task['model'].get('kappas')
                sizes.append({**prefix, 'evaluation_fold': fold, 'region': int(region),
                              'training_posts': tr[region], 'evaluation_posts': ev[region],
                              'training_share_all_posts': tr[region] / len(training),
                              'evaluation_share_all_posts': ev[region] / len(labels),
                              'evaluation_missing_region': int(ev[region] == 0),
                              'training_below_reference_count_threshold': int(tr[region] < minimum),
                              'evaluation_below_reference_count_threshold': int(ev[region] < minimum),
                              'reference_count_threshold': minimum,
                              'vmf_effective_training_mass': float(weights[region] * len(training)) if weights is not None and region < len(weights) else None,
                              'vmf_concentration': float(kappas[region]) if kappas is not None and region < len(kappas) else None,
                              'event_window_posts': None, 'event_window_support_status': 'unavailable'})
            selected = select_examples(ids, labels, pred.get(f'evaluation_centrality_{fold}'), pred.get(f'evaluation_confidence_{fold}'), seed=config.get('sampling', {}).get('seed', 20260914))
            for row in selected:
                aid = row['annotation_id']
                metadata = posts.get(aid, {})
                text = _validate_text(aid, posts, texts)
                examples.append({**prefix, 'evaluation_fold': fold, **row,
                                 'post_id': metadata.get('post_id'), 'subreddit': metadata.get('subreddit'),
                                 'year_month': metadata.get('year_month'), 'text_sha256': metadata.get('text_sha256'),
                                 'text_status': 'authenticated_original' if text is not None else 'unavailable',
                                 'text': text, 'human_region_description': '', 'human_other_explanations': '',
                                 'human_interpretability_notes': ''})
        review_regions.extend({**prefix, **row} for row in _review_summary(task, training, np.asarray(all_eval, dtype=int)))
    for a, b in combinations(tasks, 2):
        kind = comparison_kind(a['spec'], b['spec'])
        if not kind:
            continue
        for fold in sorted(set(a['spec']['eval_folds']) & set(b['spec']['eval_folds'])):
            x, y = _aligned_predictions(a, b, fold)
            summary, regions, edges = compare_partitions(x, y)
            prefix = {'comparison_kind': kind, 'left_fit': a['spec']['fit_id'], 'right_fit': b['spec']['fit_id'],
                      'family': a['spec']['family'], 'base': a['spec']['base'], 'evaluation_fold': fold,
                      'left_train_folds': '+'.join(a['spec']['train_folds']), 'right_train_folds': '+'.join(b['spec']['train_folds']),
                      'left_training_posts': len(a['predictions']['training_ids']), 'right_training_posts': len(b['predictions']['training_ids'])}
            comparison_rows.append({**prefix, **summary})
            overlap_regions.extend({**prefix, **row} for row in regions)
            edges_out.extend({**prefix, **row} for row in edges)
    outputs = {
        'fit_summary.csv': summaries, 'region_sizes.csv': sizes,
        'partition_stability.csv': [row for row in comparison_rows if row['comparison_kind'] != 'nested_training_size_consistency'],
        'sample_size_comparison.csv': [row for row in comparison_rows if row['comparison_kind'] == 'nested_training_size_consistency'],
        'region_stability.csv': overlap_regions,
        'region_overlap_edges.csv': edges_out, 'region_review_summary.csv': review_regions,
        'review_assignments_restricted.csv': reviews_out, 'representative_posts_restricted.csv': examples,
        'pending_tasks.csv': pending,
    }
    for filename, rows in outputs.items():
        _write_csv(output / filename, rows)
    result = {'status': 'complete' if not pending else 'partial', 'completed_fits': len(tasks),
              'planned_fits': len(specs), 'pending_fits': len(pending),
              'comparisons': dict(Counter(r['comparison_kind'] for r in comparison_rows)),
              'region_example_rows': len(examples), 'authenticated_text_example_rows': sum(r['text_status'] == 'authenticated_original' for r in examples),
              'human_interpretability_assessment': 'pending_manual_review',
              'event_window_support': 'unavailable', 'new_human_scores': 0, 'availability_effects_estimated': False,
              'output_sha256': {name: _sha256(output / name) for name in outputs}}
    _write_report(output, result, summaries, comparison_rows)
    result['output_sha256']['RESULTS.md'] = _sha256(output / 'RESULTS.md')
    _write_json(output / 'report_status.json', result)
    return result


def _number(value, decimals=3):
    return 'unavailable' if value is None else f'{value:.{decimals}f}'


def _write_report(output, result, summaries, comparisons):
    lines = ['# Semantic-region measurement pilot', '',
             f"Status: **{result['status']}** — {result['completed_fits']} of {result['planned_fits']} planned fits completed.", '',
             'The three balanced reference folds separate fitting from evaluation. Single-fold fits are compared on their common unseen third fold. Two-fold fits are evaluated on the remaining fold; comparison with either constituent fit measures consistency under a larger, overlapping training sample.', '',
             '| Method configuration | Training posts | Evaluation runs | Median coverage | Median regions | Converged fits | Median fit seconds |',
             '|---|---:|---:|---:|---:|---:|---:|']
    grouped = defaultdict(list)
    for row in summaries:
        grouped[(row['base'], row['training_posts'])].append(row)
    for (base, training_posts), rows in sorted(grouped.items()):
        fits = {row['fit_id']: row for row in rows}
        coverages = [row['evaluation_coverage'] for row in rows if row.get('evaluation_coverage') is not None]
        regions = [row['evaluation_regions'] for row in rows if row.get('evaluation_regions') is not None]
        seconds = [row['seconds'] for row in fits.values() if row.get('seconds') is not None]
        conv = sum(row.get('converged') in (True, 'deterministic') for row in fits.values())
        lines.append(f"| {base} | {training_posts:,} | {len(rows)} | {np.median(coverages):.1%} | {_number(float(np.median(regions)), 0)} | {conv}/{len(fits)} | {_number(float(np.median(seconds)), 1) if seconds else 'unavailable'} |" if coverages and regions else f'| {base} | {training_posts:,} | {len(rows)} | unavailable | unavailable | {conv}/{len(fits)} | unavailable |')
    lines += ['', '| Comparison | Configuration | Comparisons | Median joint coverage | Median nondegenerate ARI | Comparisons with warnings |',
              '|---|---|---:|---:|---:|---:|']
    comparison_groups = defaultdict(list)
    for row in comparisons:
        comparison_groups[(row['comparison_kind'], row['base'])].append(row)
    for (kind, base), rows in sorted(comparison_groups.items()):
        aris = [row['adjusted_rand_joint_assigned'] for row in rows if row['adjusted_rand_joint_assigned'] is not None]
        lines.append(f"| {kind.replace('_', ' ')} | {base} | {len(rows)} | {np.median([r['joint_assignment_fraction'] for r in rows]):.1%} | {_number(float(np.median(aris))) if aris else 'undefined'} | {sum(bool(r['degenerate_flags']) for r in rows)} |")
    cap_rows = [row for row in summaries if row.get('concentration_cap_hits', 0)]
    lines += ['', '## How to assess the results', '',
              '`fit_summary.csv` gives coverage, convergence, runtime and available model diagnostics for each fit/evaluation fold. Rejected posts are reported as rejected, rather than treated as one extra semantic region. ARI is calculated only on jointly assigned posts and is left undefined for all-rejected or single-region joint partitions. Read it together with coverage and warnings.', '',
              '`region_stability.csv` gives each region’s best and one-to-one Jaccard match in both directions, including members rejected by the other map. `region_overlap_edges.csv` lists all nonzero overlaps, with each direction’s fraction, so potential splits and merges remain visible. Numeric region IDs are local to each fit. A match is an empirical correspondence on this evaluation sample, not a semantic identity claim.', '',
              '`region_sizes.csv` lists training/evaluation counts and flags regions below a descriptive count threshold or absent from evaluation. That threshold is not a guarantee of statistical reliability. vMF rows include effective training mass from mixture weights, hard assignment counts, and concentration where available. Mixture components are not automatically density peaks or meaningful topics.', '',
              f"There are {len(cap_rows)} evaluation summaries with one or more vMF concentration-cap hits (a fit may occur in more than one summary). Check these and convergence before interpreting components. A cap may constrain cohesive content; it is not evidence of an AI effect.", '',
              '`partition_stability.csv` separates independent-fit reproducibility and seed stability; `sample_size_comparison.csv` contains only nested training-size comparisons. The 20k-versus-40k comparisons share training posts and cannot establish independent reproducibility or prove improvement. For density, retaining a fixed neighbor count while increasing the training size also changes the neighbor-count-to-sample-size ratio and therefore the smoothing scale. This sensitivity belongs in the interpretation of any recovered smaller regions.', '',
              f"`representative_posts_restricted.csv` contains {result['region_example_rows']} selected post examples, of which {result['authenticated_text_example_rows']} include hash-verified original text. Each available region packet draws up to three deterministic random examples, three cosine-central examples, and two lower-confidence examples where that method supplies a score; duplicates are removed. The random selection is drawn before central and confidence examples. A lower-confidence tail is not a calibrated boundary or comparable uncertainty across methods. These examples require manual reading; no interpretability winner has been selected.", '',
              '`review_assignments_restricted.csv` and `region_review_summary.csv` preserve Derek’s and Arya’s original personal-context scores separately and identify reviews used in fitting. These historically selected reviews are not a random sample of each new region. Their means and agreement describe the reviewed subset, not population personal-context prevalence. Personal context is one interpretation tool; use the blank description/other-explanation fields to consider other content patterns. No new ratings were generated.', '',
              '## Limits and next decision', '',
              'Event-window sample sizes, posting rates, regional shares during availability events, and AI effects are **unavailable**, not zero. This reference-data pilot cannot estimate those outcomes. The balanced community design describes this reference sample rather than Reddit-wide prevalence. Split rotations reuse data, so their summaries are not independent replications or confidence intervals.', '',
              'Use per-region reproducibility, coverage, convergence and manual interpretation together to shortlist a method and settings. Inspect failures and smaller regions before deciding whether additional fitting data help. Only after that decision should the chosen map be fitted on all 60,000 reference posts, checked, frozen, and applied to separate availability-event data. Discovery-selected responses must later be tested on held-out events.']
    if result['pending_fits']:
        lines += ['', f"**Partial run:** {result['pending_fits']} fits remain missing, running or failed. See `pending_tasks.csv`; comparisons currently use only completed, authenticated artifacts."]
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = report_run(args.output)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'Cannot generate three-fold report: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
