#!/usr/bin/env python3
"""Prepare a blinded, descriptive review packet from four frozen semantic maps.

This reads completed artifacts only. It does not fit models, select a winner,
call an embedding service, or estimate availability effects. Private outputs
belong on the same authorized research storage as the source posts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_semantic_threefold_pilot as pilot
import semantic_threefold_report as report

DEFAULT_SEED = 20260915
DEFAULT_PER_REGION = 25
NOTE_FIELDS = ('map_alias', 'region_alias', 'sample_posts', 'short_label',
               'what_this_region_captures', 'coherence_and_consistency',
               'mixed_subject_matter', 'uncertainty_or_unclear_boundaries',
               'other_possible_explanations', 'additional_examples_needed',
               'reviewer', 'reviewed_at')


def read_json(path):
    return json.loads(Path(path).read_text())


def rank_hash(seed, annotation_id):
    """One common uniform hash ranking, reused across all region partitions."""
    return hashlib.sha256(f'{seed}|{annotation_id}'.encode('utf-8')).hexdigest()


def alias_hash(seed, namespace, value):
    return hashlib.sha256(f'{seed}|{namespace}|{value}'.encode('utf-8')).hexdigest()


def select_candidate_specs(specs):
    """A convention fixed before content review; never a quality ranking."""
    selected = []
    for family in ('spherical_kmeans', 'vmf'):
        for k in (10, 20):
            matches = [spec for spec in specs if spec.get('family') == family and spec.get('k') == k
                       and spec.get('seed') == 11 and spec.get('train_folds') == ['A', 'B']
                       and spec.get('eval_folds') == ['C']]
            if len(matches) != 1:
                raise ValueError('Expected exactly one AB / seed11 candidate for each family and K')
            selected.append(matches[0])
    if len({spec['task_id'] for spec in selected}) != 4 or len({spec['fit_id'] for spec in selected}) != 4:
        raise ValueError('Selected candidates must have distinct task and fit identities')
    return selected


def select_region_samples(tasks, *, seed=DEFAULT_SEED, per_region=DEFAULT_PER_REGION):
    """Return internal selected rows, blind map keys, and per-region coverage.

    Selection reads only frozen IDs and hard assignments. Scores, confidence,
    centrality, historical review presence, and raw text cannot affect it.
    """
    pilot.positive_integer(per_region, 'per_region')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Seed must be an integer')
    specs = select_candidate_specs([task['spec'] for task in tasks])
    if len(tasks) != 4:
        raise ValueError('Pass exactly the four predeclared candidate maps')
    by_id = {task['spec']['task_id']: task for task in tasks}
    randomized = sorted(specs, key=lambda spec: (alias_hash(seed, 'map-alias', spec['fit_id']), spec['fit_id']))
    map_keys, region_keys, selected = [], [], []
    common_ids = None
    seen_aliases = {}
    for position, spec in enumerate(randomized):
        map_alias = f'Map {chr(65 + position)}'
        task = by_id[spec['task_id']]
        prediction = task['predictions']
        ids = np.asarray(prediction['evaluation_ids_C']).astype(str)
        labels = np.asarray(prediction['evaluation_labels_C'])
        if ids.ndim != 1 or labels.ndim != 1 or len(ids) != len(labels) or not len(ids):
            raise ValueError('Evaluation IDs and labels must be aligned nonempty vectors')
        if len(set(ids)) != len(ids) or any(not aid for aid in ids):
            raise ValueError('Evaluation IDs must be unique and nonempty')
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= spec['k']):
            raise ValueError('Directional evaluation labels must be integers in the fitted component range')
        current_ids = set(ids.tolist())
        if common_ids is None:
            common_ids = current_ids
        elif current_ids != common_ids:
            raise ValueError('All four maps must use the same evaluation post universe')
        if 'training_ids' in prediction and current_ids & set(np.asarray(prediction['training_ids']).astype(str)):
            raise ValueError('Training and evaluation posts overlap')
        map_keys.append({'map_alias': map_alias, 'task_id': spec['task_id'], 'fit_id': spec['fit_id'],
                         'family': spec['family'], 'k': spec['k'], 'seed': spec['seed'],
                         'training_folds': 'AB', 'evaluation_fold': 'C', 'evaluation_posts': len(ids)})
        original_regions = sorted(range(spec['k']), key=lambda region: (
            alias_hash(seed, 'region-alias', f"{spec['fit_id']}|{region}"), region))
        for j, original_region in enumerate(original_regions, start=1):
            region_alias = f'Region {j:02d}'
            members = ids[labels == original_region].tolist()
            chosen = sorted(members, key=lambda aid: (rank_hash(seed, aid), aid))[:per_region]
            region_keys.append({'map_alias': map_alias, 'region_alias': region_alias, 'task_id': spec['task_id'],
                                'fit_id': spec['fit_id'], 'original_region': original_region,
                                'evaluation_region_posts': len(members), 'sample_posts': len(chosen),
                                'sampling_fraction': len(chosen) / len(members) if members else 0.,
                                'all_region_posts_included': bool(members) and len(chosen) == len(members),
                                'evaluation_region_empty': not members})
            for display_order, aid in enumerate(chosen, start=1):
                post_alias = 'Post ' + alias_hash(seed, 'post-alias', aid)[:12]
                if post_alias in seen_aliases and seen_aliases[post_alias] != aid:
                    raise ValueError('Blind post alias collision')
                seen_aliases[post_alias] = aid
                selected.append({'map_alias': map_alias, 'region_alias': region_alias, 'post_alias': post_alias,
                                 'display_order': display_order, 'annotation_id': aid, 'task_id': spec['task_id'],
                                 'original_region': original_region, 'selection_rank_sha256': rank_hash(seed, aid)})
    return selected, map_keys, region_keys


def authenticate_run(results_dir, data_root):
    results = Path(results_dir).resolve()
    config, committed = read_json(results / 'config.json'), read_json(results / 'inputs.json')
    report_status = read_json(results / 'report_status.json')
    specs = read_json(results / 'tasks.json')
    if report_status.get('status') != 'complete' or report_status.get('pending_fits') != 0:
        raise ValueError('Review requires a completed source experiment')
    if report_status.get('completed_fits') != len(specs) or report_status.get('planned_fits') != len(specs):
        raise ValueError('Completed report and task inventory disagree')
    original_commit = dict(committed)
    fingerprint = original_commit.pop('input_fingerprint')
    if pilot.digest(original_commit) != fingerprint or pilot.digest(config) != committed['config_sha256']:
        raise ValueError('Source input or configuration commitment is invalid')
    if report_status.get('input_fingerprint') != fingerprint:
        raise ValueError('Completed report belongs to different source inputs')
    inputs = pilot.load_inputs(config, data_root)
    if inputs['input_sha256'] != committed['input_sha256']:
        raise ValueError('Canonical input hashes differ from the completed experiment')
    actual_fold_rows = [{key: str(value) for key, value in row.items()} for row in pilot.fold_rows(inputs)]
    if pilot.read_csv(results / 'fold_assignments_restricted.csv') != actual_fold_rows:
        raise ValueError('Source fold assignments differ from the authenticated inputs')
    source_code = results.parent / 'code'
    for name, expected in committed['code_sha256'].items():
        if Path(name).name != name or pilot.sha256(source_code / name) != expected:
            raise ValueError('Frozen source implementation differs from its committed hash')
    for module in (pilot, report):
        path = Path(module.__file__)
        if committed['code_sha256'].get(path.name) != pilot.sha256(path):
            raise ValueError('Imported helper differs from the frozen source implementation')
    for spec in specs:
        status = read_json(results / 'tasks' / f"task_{spec['task_id']:03d}" / 'status.json')
        if (status.get('status') != 'complete' or status.get('task_id') != spec['task_id']
                or status.get('fit_id') != spec['fit_id'] or status.get('input_fingerprint') != fingerprint
                or status.get('config_sha256') != committed['config_sha256']):
            raise ValueError('Source task status differs from the completed authenticated experiment')
    chosen_specs = select_candidate_specs(specs)
    tasks, pending = report._authenticated_tasks(results, chosen_specs)
    if pending or len(tasks) != 4:
        raise ValueError('Selected candidate artifacts are incomplete')
    expected_train = inputs['folds']['A'] + inputs['folds']['B']
    expected_evaluation = inputs['folds']['C']
    for task in tasks:
        prediction, fit = task['predictions'], task['fit']
        if (prediction['training_ids'].astype(str).tolist() != expected_train
                or prediction['evaluation_ids_C'].astype(str).tolist() != expected_evaluation):
            raise ValueError('Selected map uses different training or evaluation IDs/order')
        if not fit['model_scalars'].get('converged'):
            raise ValueError('Selected directional map did not reach numerical convergence')
        reviews = {row['annotation_id']: row for row in task['reviews']}
        if len(reviews) != len(task['reviews']) or set(reviews) != set(inputs['reviews']):
            raise ValueError('Existing review assignments do not cover the original annotation set')
        for aid, original in inputs['reviews'].items():
            for rater in ('derek', 'arya'):
                field = f'{rater}_personal_req_score'
                if reviews[aid].get(field) != original[field]:
                    raise ValueError('An existing personal-context annotation was modified')
    metadata_hashes = {name: pilot.sha256(results / name) for name in
                       ('config.json', 'inputs.json', 'tasks.json', 'fold_assignments_restricted.csv', 'report_status.json')}
    selected_artifacts = {}
    for spec in chosen_specs:
        directory = results / 'tasks' / f"task_{spec['task_id']:03d}"
        status = read_json(directory / 'status.json')
        selected_artifacts[str(spec['task_id'])] = dict(status['artifact_sha256'], **{'status.json': pilot.sha256(directory / 'status.json')})
    return config, inputs, tasks, {'source_results': str(results), 'input_fingerprint': fingerprint,
                                  'metadata_sha256': metadata_hashes, 'selected_artifact_sha256': selected_artifacts,
                                  'frozen_source_code_sha256': committed['code_sha256'], 'input_sha256': inputs['input_sha256']}


def load_selected_texts(inputs, selected_ids):
    """Stream the authenticated source, retaining text only for selected IDs."""
    wanted = set(selected_ids)
    if not wanted.issubset(inputs['index']):
        raise ValueError('Selected post ID is outside the authenticated embedding corpus')
    if 'texts' not in inputs['paths']:
        raise ValueError('An authenticated source text file is required')
    path = inputs['paths']['texts']
    if pilot.sha256(path) != inputs['input_sha256']['texts']:
        raise ValueError('Source text file changed after selection policy was recorded')
    texts = {}
    with path.open(encoding='utf-8', newline='') as stream:
        rows = (json.loads(line) for line in stream if line.strip()) if path.suffix == '.jsonl' else csv.DictReader(stream)
        for row in rows:
            aid = row.get('annotation_id')
            if aid not in wanted:
                continue
            if aid in texts:
                raise ValueError('Selected post appears more than once in the source text file')
            expected = inputs['index'][aid]
            for key in ('post_id', 'text_sha256'):
                if row.get(key) and row[key] != expected[key]:
                    raise ValueError('Selected source text identity differs from the embedded post')
            text = report._validate_text(aid, inputs['index'], {aid: row})
            if text is None:
                raise ValueError('Selected post has no original text')
            texts[aid] = text
    if set(texts) != wanted:
        raise ValueError('Some selected posts are missing from the authenticated source text')
    # Detect concurrent modification during streaming as well as before it.
    if pilot.sha256(path) != inputs['input_sha256']['texts']:
        raise ValueError('Source text file changed while the review packet was being built')
    return texts


def render_reader(sample_rows, region_rows):
    """Render supplied blind rows as inert text, with no scripts or resources."""
    by_region = defaultdict(list)
    for row in sample_rows:
        by_region[(row['map_alias'], row['region_alias'])].append(row)
    groups = defaultdict(list)
    for row in region_rows:
        groups[row['map_alias']].append(row)
    escape = lambda value: html.escape(str(value), quote=True)
    fragments = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
                 '<meta name="viewport" content="width=device-width,initial-scale=1">',
                 '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; style-src &#39;unsafe-inline&#39;; img-src &#39;none&#39;; base-uri &#39;none&#39;; form-action &#39;none&#39;">',
                 '<title>Semantic region reading packet</title><style>',
                 'body{margin:0;color:#20252b;background:#f8fafb;font:16px/1.55 system-ui,sans-serif}',
                 'nav{position:fixed;inset:0 auto 0 0;width:205px;overflow:auto;padding:24px;background:#eaf0f3}',
                 'nav a{display:block;color:#17475c;padding:4px 0;text-decoration:none}nav h2{font-size:17px;margin:20px 0 6px}',
                 'main{max-width:930px;margin-left:253px;padding:30px 40px}h1{font-size:30px}h2{font-size:24px}',
                 'section{scroll-margin-top:20px;margin:36px 0 60px}article{padding:18px 22px;margin:16px 0;background:white;border:1px solid #d9e1e6;border-radius:8px}',
                 '.text{white-space:pre-wrap;overflow-wrap:anywhere}.post-title{font-size:14px;font-weight:600;color:#4a5a64}',
                 '.guide{padding:18px;background:#edf4f2;border-left:4px solid #517a70}.muted{color:#52616a}',
                 '@media(max-width:760px){nav{position:static;width:auto}main{margin:0;padding:20px}nav a{display:inline-block;margin-right:12px}}',
                 '@media print{nav{display:none}main{margin:0;padding:0}section{break-before:page}article{break-inside:avoid}}',
                 '</style></head><body><nav aria-label="Region navigation"><a href="#start">Instructions</a>']
    anchors = {}
    for map_number, (map_alias, regions) in enumerate(sorted(groups.items()), start=1):
        fragments.append(f'<h2>{escape(map_alias)}</h2>')
        for region_number, region in enumerate(regions, start=1):
            anchor = f'm{map_number}-r{region_number}'
            anchors[(map_alias, region['region_alias'])] = anchor
            fragments.append(f'<a href="#{anchor}">{escape(region["region_alias"])}</a>')
    fragments += ['</nav><main><header id="start"><h1>Semantic region reading packet</h1>',
                  '<p>Read the sampled posts, then record your observations in <strong>region_notes.csv</strong>. '
                  'Use the map and region codes to connect your notes with this reader. This page does not save annotations.</p>',
                  '<div class="guide"><strong>For each region:</strong> What does it capture? How consistently do the posts fit together? '
                  'Are there mixed subjects or unclear boundaries? What other explanations could account for the grouping? '
                  'Note uncertainty and whether more examples are needed. You do not need to assign a particular personal-context label.</div>',
                  '<p class="muted">Examples were selected by a fixed random ranking within each region, without using text, '
                  'prior scores, or model diagnostics. Repeated post codes identify the same post across maps. '
                  'Identities are coded; the content and number of regions may still suggest differences between maps. '
                  'Keep the separate interpretation key closed during this initial reading.</p></header>']
    for map_alias, regions in sorted(groups.items()):
        for region in regions:
            region_alias = region['region_alias']
            rows = by_region[(map_alias, region_alias)]
            fragments.append(f'<section id="{anchors[(map_alias, region_alias)]}"><h2>{escape(map_alias)} · {escape(region_alias)}</h2>')
            fragments.append(f'<p class="muted">{len(rows)} sampled posts</p>')
            if not rows:
                fragments.append('<p>No evaluation posts were assigned to this region. Leave content judgments blank.</p>')
            for row in rows:
                fragments.append(f'<article><div class="post-title">{escape(row["display_order"])} · {escape(row["post_alias"])}</div>')
                fragments.append(f'<div class="text">{escape(row["text"])}</div></article>')
            fragments.append('<p><a href="#start">Back to instructions</a></p></section>')
    fragments.append('</main></body></html>')
    return '\n'.join(fragments) + '\n'


def build_packet(results_dir, data_root, output, *, seed=DEFAULT_SEED, per_region=DEFAULT_PER_REGION):
    output, results = Path(output).resolve(), Path(results_dir).resolve()
    if output == results or results in output.parents or output in results.parents:
        raise ValueError('The review packet must have a separate new output directory')
    if output.exists():
        raise FileExistsError('Review output already exists; refusing to overwrite the packet or its notes')
    config, inputs, tasks, source = authenticate_run(results, data_root)
    selections, map_keys, region_keys = select_region_samples(tasks, seed=seed, per_region=per_region)
    policy = {'schema_version': 1, 'purpose': 'Initial descriptive reading; no winner selection or availability inference.',
              'candidate_convention': {'training_folds': ['A', 'B'], 'evaluation_fold': 'C', 'seed': 11,
                                       'families': ['spherical_kmeans', 'vmf'], 'components': [10, 20]},
              'candidate_selection_basis': 'Fixed convenience convention before reading content, not a performance ranking.',
              'sampling': {'seed': seed, 'posts_per_region': per_region, 'algorithm': 'Ascending SHA256(UTF8(str(seed) + "|" + annotation_id)); first min(n, requested) per region.',
                           'same_post_ranking_across_maps': True, 'empty_regions_retained': True,
                           'unused_for_selection': ['post text', 'historical ratings', 'confidence', 'centrality', 'stability']},
              'blinding': {'map_aliases': 'Fit IDs sorted by a separate seeded hash, assigned Map A through Map D.',
                           'region_aliases': 'Region IDs independently sorted by a separate seeded hash within each map.',
                           'scope': 'Metadata labels hidden; content and region counts may still suggest map differences.'},
              'source_input_fingerprint': source['input_fingerprint'], 'source_metadata_sha256': source['metadata_sha256'],
              'selected_rows_sha256': pilot.digest(selections), 'recorded_before_selected_text_loading': True}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + '.building.' + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    try:
        (staging / 'reviewer').mkdir(mode=0o700)
        (staging / 'restricted_key').mkdir(mode=0o700)
        # Commit the data-independent selection before retaining any selected text.
        pilot.write_json(staging / 'selection_policy.json', policy)
        pilot.write_csv(staging / 'restricted_key' / 'map_key.csv', map_keys)
        pilot.write_csv(staging / 'restricted_key' / 'region_key.csv', region_keys)
        selected_ids = {row['annotation_id'] for row in selections}
        texts = load_selected_texts(inputs, selected_ids)
        sample_rows = [{key: row[key] for key in ('map_alias', 'region_alias', 'post_alias', 'display_order')}
                       | {'text': texts[row['annotation_id']]} for row in selections]
        note_rows = [{name: region.get(name, '') for name in NOTE_FIELDS} for region in region_keys]
        pilot.write_csv(staging / 'reviewer' / 'sampled_posts_restricted.csv', sample_rows,
                        fields=['map_alias', 'region_alias', 'post_alias', 'display_order', 'text'])
        pilot.write_csv(staging / 'reviewer' / 'region_notes.csv', note_rows, fields=NOTE_FIELDS)
        post_keys = [dict(row, post_id=inputs['index'][row['annotation_id']]['post_id'],
                          text_sha256=inputs['index'][row['annotation_id']]['text_sha256']) for row in selections]
        pilot.write_csv(staging / 'restricted_key' / 'sampled_post_key.csv', post_keys)
        region_aliases = {(row['task_id'], row['original_region']): (row['map_alias'], row['region_alias']) for row in region_keys}
        sampled_pairs = {(row['task_id'], row['annotation_id']) for row in selections}
        rating_rows = []
        for task in tasks:
            task_id = task['spec']['task_id']
            for assignment in task['reviews']:
                aid, region = assignment['annotation_id'], int(assignment['region'])
                if (task_id, region) not in region_aliases:
                    raise ValueError('Historical review has an invalid region assignment')
                map_alias, region_alias = region_aliases[(task_id, region)]
                rating_rows.append(dict(inputs['reviews'][aid], map_alias=map_alias, region_alias=region_alias,
                                        post_alias='Post ' + alias_hash(seed, 'post-alias', aid)[:12],
                                        task_id=task_id, original_region=region,
                                        used_in_fit=assignment['used_in_fit'], fold=assignment['fold'],
                                        included_in_random_reading_packet=int((task_id, aid) in sampled_pairs),
                                        scope='Historically selected annotation subset; optional interpretation after initial reading.'))
        pilot.write_csv(staging / 'restricted_key' / 'existing_ratings_context.csv', rating_rows,
                        fields=None if rating_rows else ['map_alias', 'region_alias', 'post_alias'])
        reader = render_reader(sample_rows, note_rows)
        (staging / 'reviewer' / 'reader.html').write_text(reader, encoding='utf-8')
        (staging / 'reviewer' / 'README.md').write_text(
            '# Read the regions\n\nOpen `reader.html` and add your observations to `region_notes.csv`. '
            'The page displays examples but does not save notes. Keep the separate `restricted_key` directory closed during the initial reading.\n\n'
            'Summarize what each group captures, whether its posts fit together, mixed subjects, unclear boundaries, and other explanations. '
            'State uncertainty and whether you need more examples. No particular personal-context label is required. '
            'Repeated post codes indicate the same post across maps, so you can reuse your reading of that post.\n\n'
            'The examples were sampled without using content or prior scores. A small reading packet can identify interpretation problems; '
            'it does not establish a definitive label, select a final map, or measure an AI effect.\n', encoding='utf-8')
        sampling_rows = [{key: row[key] for key in ('map_alias', 'region_alias', 'evaluation_region_posts', 'sample_posts',
                                                   'sampling_fraction', 'all_region_posts_included', 'evaluation_region_empty')}
                         for row in region_keys]
        pilot.write_csv(staging / 'sampling_summary.csv', sampling_rows)
        manifest = {'schema_version': 1, 'status': 'prepared_for_manual_review', 'created_utc': pilot.now(),
                    'source': source, 'packet_code_sha256': pilot.sha256(Path(__file__)),
                    'selection_policy_sha256': pilot.sha256(staging / 'selection_policy.json'),
                    'counts': {'candidate_maps': len(map_keys), 'regions': len(region_keys),
                               'nonempty_regions': sum(not row['evaluation_region_empty'] for row in region_keys),
                               'sampled_post_appearances': len(selections), 'unique_sampled_posts': len(selected_ids),
                               'evaluation_posts_per_map': len(inputs['folds']['C']),
                               'regions_with_all_evaluation_posts_sampled': sum(row['all_region_posts_included'] for row in region_keys),
                               'existing_review_posts': len(inputs['reviews']), 'existing_rating_mapping_rows': len(rating_rows)},
                    'per_region_sampling': sampling_rows, 'new_human_scores': 0, 'map_selected_or_refitted': False,
                    'event_window_support': 'unavailable', 'availability_effects_estimated': False,
                    'manual_review_status': 'not_started',
                    'scope': 'Descriptive reference-data review; later method selection and event-window adequacy require separate assessment.',
                    'output_sha256': {str(path.relative_to(staging)): pilot.sha256(path) for path in sorted(staging.rglob('*')) if path.is_file()}}
        pilot.write_json(staging / 'manifest.json', manifest)
        os.rename(staging, output)
        return {key: manifest[key] for key in ('status', 'counts', 'manual_review_status', 'event_window_support')}
    except BaseException:
        shutil.rmtree(staging)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--per-region', type=int, default=DEFAULT_PER_REGION)
    args = parser.parse_args(argv)
    try:
        result = build_packet(args.results_dir, args.data_root, args.output, seed=args.seed, per_region=args.per_region)
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        # Error type only: no private post text or identifiers in terminal logs.
        print(f'Review preparation stopped: {type(exc).__name__}. Check the authenticated inputs and output location.', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
